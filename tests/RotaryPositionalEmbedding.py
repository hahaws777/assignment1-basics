import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # positions = 0,1,...,max_seq_len-1
        pos = torch.arange(max_seq_len, dtype=torch.float, device=device).unsqueeze(1)
        freqs = theta ** (-torch.arange(0, d_k, 2, dtype=torch.float, device=device) / d_k)
        angles = pos * freqs
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        self.register_buffer("cos_angles", cos_angles, persistent=False)
        self.register_buffer("sin_angles", sin_angles, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos_angles = self.cos_angles[token_positions]
        sin_angles = self.sin_angles[token_positions]
        output_even = x_even * cos_angles - x_odd * sin_angles
        output_odd = x_even * sin_angles + x_odd * cos_angles
        # 交替写入output
        output = torch.empty_like(x)
        output[..., 0::2] = output_even
        output[..., 1::2] = output_odd

        return output
