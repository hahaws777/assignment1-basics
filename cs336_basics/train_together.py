from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from tests.adapters import (
    get_adamw_cls,
    run_cross_entropy,
    run_get_batch,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    run_load_checkpoint,
    run_save_checkpoint,
    run_transformer_lm,
)


class TransformerLayerParams(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        scale = 0.02
        self.q_proj = nn.Parameter(torch.randn(d_model, d_model) * scale)
        self.k_proj = nn.Parameter(torch.randn(d_model, d_model) * scale)
        self.v_proj = nn.Parameter(torch.randn(d_model, d_model) * scale)
        self.output_proj = nn.Parameter(torch.randn(d_model, d_model) * scale)

        self.ffn_w1 = nn.Parameter(torch.randn(d_ff, d_model) * scale)
        self.ffn_w2 = nn.Parameter(torch.randn(d_model, d_ff) * scale)
        self.ffn_w3 = nn.Parameter(torch.randn(d_ff, d_model) * scale)

        self.ln1 = nn.Parameter(torch.ones(d_model))
        self.ln2 = nn.Parameter(torch.ones(d_model))


class AdapterTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        scale = 0.02
        self.token_embeddings = nn.Parameter(torch.randn(vocab_size, d_model) * scale)
        self.layers = nn.ModuleList(
            [TransformerLayerParams(d_model=d_model, d_ff=d_ff) for _ in range(num_layers)]
        )
        self.ln_final = nn.Parameter(torch.ones(d_model))
        self.lm_head = nn.Parameter(torch.randn(vocab_size, d_model) * scale)

    def _state_dict_for_adapter(self) -> dict[str, torch.Tensor]:
        d: dict[str, torch.Tensor] = {
            "token_embeddings.weight": self.token_embeddings,
            "ln_final.weight": self.ln_final,
            "lm_head.weight": self.lm_head,
        }
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}"
            d[f"{prefix}.attn.q_proj.weight"] = layer.q_proj
            d[f"{prefix}.attn.k_proj.weight"] = layer.k_proj
            d[f"{prefix}.attn.v_proj.weight"] = layer.v_proj
            d[f"{prefix}.attn.output_proj.weight"] = layer.output_proj
            d[f"{prefix}.ln1.weight"] = layer.ln1
            d[f"{prefix}.ln2.weight"] = layer.ln2
            d[f"{prefix}.ffn.w1.weight"] = layer.ffn_w1
            d[f"{prefix}.ffn.w2.weight"] = layer.ffn_w2
            d[f"{prefix}.ffn.w3.weight"] = layer.ffn_w3
        return d

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        return run_transformer_lm(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            weights=self._state_dict_for_adapter(),
            in_indices=in_indices,
        )


def load_token_array(path: Path, dtype: str) -> np.ndarray:
    # For quick large-data experiments, allow raw text files and treat UTF-8 bytes
    # as token IDs in [0, 255] without loading the full file into memory.
    if path.suffix == ".txt":
        return np.memmap(path, mode="r", dtype=np.uint8)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    return np.memmap(path, mode="r", dtype=np.dtype(dtype))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    vocab_size: int,
    eval_steps: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_steps):
        x, y = run_get_batch(val_data, batch_size=batch_size, context_length=context_length, device=device)
        logits = model(x)
        loss = run_cross_entropy(
            inputs=logits.reshape(-1, vocab_size),
            targets=y.reshape(-1),
        )
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LM with adapters and checkpointing.")
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--valid_data", type=Path, required=True)
    parser.add_argument("--data_dtype", type=str, default="uint16")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--checkpoint_path", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.context_length <= 0:
        raise ValueError("context_length must be positive")
    if args.max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    train_data = load_token_array(args.train_data, dtype=args.data_dtype)
    valid_data = load_token_array(args.valid_data, dtype=args.data_dtype)

    model = AdapterTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)

    AdamW = get_adamw_cls()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume and args.checkpoint_path.exists():
        start_iter = run_load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Resumed from checkpoint at step {start_iter}")

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for it in range(start_iter, args.max_iters):
        lr = run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.lr_max,
            min_learning_rate=args.lr_min,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = run_get_batch(train_data, batch_size=args.batch_size, context_length=args.context_length, device=args.device)
        logits = model(x)
        loss = run_cross_entropy(inputs=logits.reshape(-1, args.vocab_size), targets=y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        run_gradient_clipping(model.parameters(), max_l2_norm=args.max_grad_norm)
        optimizer.step()

        if it % args.eval_interval == 0:
            val_loss = evaluate(
                model=model,
                val_data=valid_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                vocab_size=args.vocab_size,
                eval_steps=args.eval_steps,
            )
            print(
                f"iter={it} lr={lr:.3e} train_loss={loss.item():.4f} val_loss={val_loss:.4f}"
            )

        if it > start_iter and it % args.save_interval == 0:
            run_save_checkpoint(model=model, optimizer=optimizer, iteration=it, out=args.checkpoint_path)
            print(f"checkpoint saved to {args.checkpoint_path} at iter={it}")

    run_save_checkpoint(model=model, optimizer=optimizer, iteration=args.max_iters, out=args.checkpoint_path)
    print(f"training done, final checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
