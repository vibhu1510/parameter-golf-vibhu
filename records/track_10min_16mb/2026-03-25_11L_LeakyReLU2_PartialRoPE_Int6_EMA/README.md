# 11L LeakyReLU² + Partial RoPE + Int6 + EMA

**Expected val_bpb: ~1.1200** | **<16 MB** | 8×H100 SXM

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with **LeakyReLU(0.5)²** |
| BigramHash | 2048 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Warmdown | 3500 steps |
| EMA | decay=0.997 |
| Late QAT | scale < 0.15 |

## Key Improvement: LeakyReLU(0.5)²

One-line activation change delivering ~-0.003 BPB over relu²:

```python
# Before (relu²):
x = torch.relu(self.fc(x)).square()

# After (leaky relu²):
x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
```

LeakyReLU with slope 0.5 preserves negative gradient flow through the MLP, allowing the model to learn from both positive and negative pre-activations. The squaring step still produces non-negative outputs.

## Quantization & Compression

- Int6 per-row quantization for attention and MLP weights (clip_range=31)
- Int8 per-row for embedding weights
- QAT-aware training (STE with int6 simulation) enabled when LR scale drops below 0.15
- zstd level-22 compression

## Evaluation

- Standard eval at training checkpoints
- Sliding window eval (stride=64) for final scoring

## Base

Built on PR #414 stack (11L EMA + GPTQ-lite + warmdown 3500 + QAT 0.15), scoring 1.1233.
LeakyReLU² improvement from PR #493 / PR #518.
