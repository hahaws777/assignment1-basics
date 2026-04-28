import torch
import torch.nn.functional as F


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    # (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len) --> (batch, heads, seq_len, seq_len)
    scaled = torch.matmul(q, k.transpose(-1, -2)) / (d_k ** 0.5)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, head_dim) --> (batch, heads, seq_len, head_dim)
    values = torch.matmul(attention, v)
    return values, attention

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, input_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = d_model // num_heads

        self.qkv_layer = torch.nn.Linear(input_dim, 3 * d_model)
        self.linear_layer = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, mask = None):
        batch_size, seq_len, input_dim = x.shape
        qkv = self.qkv_layer(x)
        print(f"qkv size: {qkv.shape}")

        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        print(f"qkv reshaped size: {qkv.shape}")

        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3, self.head_dim)

        qkv = qkv.permute(0, 3, 2, 1, 4)
        print(f"qkv permuted size: {qkv.shape}")

        q, k, v = qkv.unbind(dim=1)
        print(f"qkv chunked size: {q.shape}, {k.shape}, {v.shape}")

        values,attention = scaled_dot_product(q, k, v, mask)
        print(f"values size: {values.shape}, attention size: {attention.shape}")

        values = values.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        print(f"values permuted size: {values.shape}")

        out = self.linear_layer(values)
        print(f"out size: {out.shape}")
        return out


if __name__ == "__main__":

    # Model/inputs setup
    input_dim = 1024   # Input feature size per token
    d_model = 512      # Embedding/model size (must divide num_heads)
    num_heads = 8
    batch_size = 30
    sequence_length = 5

    # Create random input
    x = torch.randn((batch_size, sequence_length, input_dim))

    # Instantiate MultiheadAttention class and run
    model = MultiHeadAttention(d_model, num_heads, input_dim)
    output = model.forward(x)
