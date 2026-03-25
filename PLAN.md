# Parameter Golf Challenge - Analysis & Plan

## Challenge Summary

**Goal**: Train the best language model that fits in a **16MB artifact** (code + compressed model) and trains in **under 10 minutes on 8xH100s**, evaluated by **bits-per-byte (BPB)** compression on the FineWeb validation set.

**Current SOTA**: 1.1194 BPB (LeakyReLU² + Legal TTT + Parallel Muon)
**Baseline**: 1.2244 BPB (9L, 512d, 1024 vocab, tied embeddings)

---

## Key Constraints

| Constraint | Limit |
|---|---|
| Artifact size | 16,000,000 bytes (code + compressed model) |
| Training time | 10 minutes on 8xH100 SXM |
| Evaluation time | 10 minutes on 8xH100 SXM (separate budget) |
| Validation data during training | NOT allowed |
| External downloads at eval time | NOT allowed |
| SOTA improvement threshold | >= 0.005 nats, p < 0.01 significance |

---

## What Top Submissions Do (Techniques Stack)

### Architecture (biggest gains)
1. **11 layers** instead of 9 (more depth within budget)
2. **3x MLP expansion** instead of 2x (more capacity per layer)
3. **Exclusive Self Attention (XSA)** on last 4 layers (richer attention)
4. **Partial RoPE** (16/64 dims) - lets most dims attend position-free
5. **BigramHash embeddings** (1536-2048 buckets) - richer input representation
6. **SmearGate** - improved gating mechanism
7. **U-Net skip connections** with learned weights (already in baseline)
8. **LN Scale factor** 1/sqrt(layer+1) - stabilizes deeper networks
9. **LeakyReLU(0.5)²** activation (preserves negative gradient flow)
10. **Value Embedding (VE128)** shared across last layers

### Quantization & Compression (critical for 16MB budget)
1. **Int6 quantization** (mixed: int6 for blocks, int8 for embeddings)
2. **GPTQ-lite** per-row optimal clip percentile search (free at train time)
3. **Late QAT** (STE fake-quantization when LR < threshold)
4. **zstd-22** compression (better than zlib)
5. **lzma** compression (used by #1 submission)

### Training Optimization
1. **EMA** (decay 0.997) + **Tight SWA** (every 50 steps)
2. **Warmdown tuning** (3500 iterations vs default 1200)
3. **Muon optimizer** for matrix params (already in baseline)
4. **Parallel Muon** - batched Newton-Schulz, async comms

### Evaluation Tricks
1. **Sliding window evaluation** (stride=64, increases effective context)
2. **Test-Time Training (TTT)** - score-first, then fine-tune on scored tokens with LoRA/SGD
3. **Longer eval sequence length** (2048-4096 during eval)

---

## Proposed Implementation Plan

### Phase 1: Low-Hanging Fruit (Expected: ~1.20 -> ~1.17 BPB)

These are proven techniques from the leaderboard with clear gains:

**Step 1.1 - Increase to 11 layers**
- Change `num_layers` from 9 to 11
- Adjust U-Net encoder/decoder split accordingly
- Validates that the model still fits in 16MB with int8+zlib

**Step 1.2 - 3x MLP expansion**
- Change `mlp_mult` from 2 to 3
- Provides more capacity per layer

**Step 1.3 - Better compression: zstd-22**
- Replace zlib with zstd at compression level 22
- Frees ~0.5-1MB of artifact budget for more parameters

**Step 1.4 - Int6 quantization**
- Switch from int8 to int6 for block weight matrices
- Keep int8 for embeddings
- Frees significant artifact budget

**Step 1.5 - Sliding window evaluation**
- Implement sliding window with stride=64 at eval time
- Nearly free BPB improvement

### Phase 2: Architecture Improvements (Expected: ~1.17 -> ~1.13 BPB)

**Step 2.1 - BigramHash embeddings**
- Add bigram hash lookup (1536-2048 buckets, 128d)
- Enriches token representations with local context

**Step 2.2 - Partial RoPE (16/64 dims)**
- Only apply rotary embeddings to first 16 of 64 dims
- Lets remaining dims learn position-invariant patterns
- Proven -0.0015 BPB gain

**Step 2.3 - Exclusive Self Attention (XSA) on last 4 layers**
- Richer attention mechanism for deeper layers
- Proven -0.002+ BPB gain

**Step 2.4 - LN Scale factor**
- Apply 1/sqrt(layer_idx+1) scaling to layer norms
- Stabilizes deeper network

**Step 2.5 - LeakyReLU(0.5)² activation**
- Replace relu² with leaky_relu(0.5)²
- Eliminates dead neurons, -0.003 BPB gain

### Phase 3: Training Optimization (Expected: ~1.13 -> ~1.12 BPB)

**Step 3.1 - EMA weight averaging**
- Add exponential moving average (decay 0.997)
- Stack with existing SWA
- Proven -0.0006 BPB

**Step 3.2 - GPTQ-lite quantization**
- Post-training per-row optimal clip percentile search
- Zero training cost, -0.0006 BPB

**Step 3.3 - Late QAT**
- Enable STE int6 fake-quantization when LR scale drops below 0.15
- Trains model to be quantization-aware

**Step 3.4 - Warmdown tuning**
- Increase warmdown_iters from 1200 to 3500
- Fine-tune learning rate schedule

### Phase 4: Advanced Techniques (Expected: ~1.12 -> ~1.10 BPB)

**Step 4.1 - Test-Time Training (TTT)**
- Implement score-first sliding window TTT
- Evaluate chunk, then fine-tune, repeat
- Must ensure legality: only train on already-scored tokens
- Proven -0.0025 BPB

**Step 4.2 - Parallel Muon optimizer**
- Batch Newton-Schulz iterations via torch.bmm
- Async reduce-scatter/all-gather
- Gains training speed (more steps in 10 min)

**Step 4.3 - SmearGate**
- Improved gating mechanism for residual connections

**Step 4.4 - Value Embedding (VE128)**
- Shared value embeddings across last 2 layers

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Model exceeds 16MB after changes | Monitor artifact size after each step; int6 + zstd gives headroom |
| Training exceeds 10 min | Profile each change; Parallel Muon reclaims speed |
| Techniques don't stack | Validate each change independently with 3-seed runs |
| TTT exceeds 10 min eval budget | Budget ~5 min for TTT, tune chunk size |

## Priority Order

If time-limited, implement in this order for maximum BPB improvement per effort:
1. 11 layers + 3x MLP + int6 + zstd (biggest architectural gains)
2. Sliding window eval + Partial RoPE + LeakyReLU² (easy wins)
3. EMA + GPTQ-lite (proven free gains)
4. XSA + BigramHash (moderate complexity, good gains)
5. TTT (highest complexity, strong gains)
