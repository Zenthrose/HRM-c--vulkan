import torch
import numpy as np
import argparse
from pathlib import Path

# Configuration matching C++ AttentionConfig
hidden_size = 128
head_dim = 64
num_heads = 2
num_key_value_heads = 2
causal = False

# Example dimensions
batch_size = 1
seq_len = 256

# Generate dummy input data
# In C++, hidden_states is used to derive Q, K, V
# Here, we'll directly generate Q, K, V for simplicity, representing the output of qkv_proj split.
# These dimensions should match the buffer sizes calculated in C++ init_vulkan_objects

q_shape = (batch_size, seq_len, num_heads, head_dim)
k_shape = (batch_size, seq_len, num_key_value_heads, head_dim)
v_shape = (batch_size, seq_len, num_key_value_heads, head_dim)

# Use consistent random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

query = torch.randn(q_shape, dtype=torch.float32)
key = torch.randn(k_shape, dtype=torch.float32)
value = torch.randn(v_shape, dtype=torch.float32)

# Compute actual attention
# For simplicity, implement basic attention without FlashAttention
def simple_attention(q, k, v, causal=False):
    # q, k, v: [batch, seq, heads, dim]
    batch, seq, heads, dim = q.shape
    scale = dim ** -0.5
    # Compute scores: [batch, heads, seq, seq]
    scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * scale
    if causal:
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    # Apply to v
    output = torch.einsum('bhqk,bhkd->bhqd', attn, v)
    return output

output = simple_attention(query, key, value, causal=causal)

# Flatten tensors for saving
query_flat = query.numpy().flatten()
key_flat = key.numpy().flatten()
value_flat = value.numpy().flatten()
output_flat = output.numpy().flatten()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate attention test data')
parser.add_argument('--output-dir', type=str, default='.', help='Output directory for test data files')
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Save to text files
np.savetxt(output_dir / "test_data_query.txt", query_flat)
np.savetxt(output_dir / "test_data_key.txt", key_flat)
np.savetxt(output_dir / "test_data_value.txt", value_flat)
np.savetxt(output_dir / "test_data_output.txt", output_flat)

print(f"Test data (query, key, value, output) saved to {output_dir}/")