# SOFTS — Replacing Encoder with DINOv3 Backbone

Practical Assignment 2 Requirement 1: **Replace the SOFTS encoder with DINOv3** (the model features an encoder, preserving the original backbone architecture semantics). The decoder (`projection`) and inverted embedding (`DataEmbedding_inverted`) remain unchanged.

- **Upstream Repository:** [Secilia-Cxy/SOFTS](https://github.com/Secilia-Cxy/SOFTS) — *SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion* (NeurIPS 2024).
- **DINOv3 Backbone:** Meta AI, ViT-S/16 (`dinov3_vits16`, embedding dimension 384, 12 blocks, 6 heads, 4 register tokens, 2D Axial RoPE, LayerScale init $10^{-5}$). Architecture specifications derived from [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) (`dinov3/hub/backbones.py`).

---

## 1. Scope of Modifications

| Component | Actual Implementation | Status |
|---|---|---|
| **Backbone / Encoder** | `DINOv3Encoder` in [`layers/DINOv3Encoder.py`](layers/DINOv3Encoder.py) | **Fully Replaced (New)** |
| **Replaced Module** | `STAR` (Aggregate - Redistribute) + `EncoderLayer` | **Removed from inference path** |
| **Preserved Modules** | `DataEmbedding_inverted`, `projection` head, instance norm/denorm | **100% Retained (SOFTS standard)** |

```text
Original SOFTS Pipeline (NeurIPS 2024):
  x [B, T, C] → DataEmbedding_inverted → STAR Modules (Core Fusion) → Linear Projection → Forecast [B, H, C]

SOFTS-DINOv3 Pipeline (Proposed Architecture):
  x [B, T, C] → DataEmbedding_inverted → DINOv3 ViT-S/16 Backbone → Linear Projection → Forecast [B, H, C]
                                           ├── Adapter in_proj (d_model → 384)
                                           ├── 2D Axial RoPE Grid + 1 CLS + 4 Register Tokens
                                           ├── 12 DINOv3 Pre-norm Blocks + LayerScale
                                           └── Adapter out_proj (384 → d_model)
```

---

## 2. Modified and New Files (Update Pointers for Review)

Only two files differ from the original upstream repository. Every modified or newly added block is explicitly tagged with `[DINOv3-UPDATE]`.

| Priority | Path | Description | Key Reference Lines |
|---|---|---|---|
| **Core (New)** | [`layers/DINOv3Encoder.py`](layers/DINOv3Encoder.py) | **New file.** DINOv3 ViT-S/16 encoder with exact signature compatibility `(x, attn_mask) -> (x, attns)`, where `x` is `[B, L, d_model]`. | Entire file |
| **Core (Modified)** | [`models/SOFTS.py`](models/SOFTS.py) | **Modified.** Updated imports and initialized `self.encoder = DINOv3Encoder(...)` replacing `Encoder` + `EncoderLayer` + `STAR`. | See 3 checkpoints below |
| Base | `layers/Embed.py` | Unchanged. Copied for standalone execution. | — |
| Deprecated | `layers/Transformer_EncDec.py` | **No longer imported.** Kept for reference and parity check. | — |

### Three Key Checkpoints in `models/SOFTS.py`:
Line numbers refer to the updated version in this repository:

1. **Imports (Lines 6–7):** Removed `from layers.Transformer_EncDec import Encoder, EncoderLayer`. Added `from layers.DINOv3Encoder import DINOv3Encoder`.
2. **`Model.__init__` (Lines 62–95, commented `[DINOv3-UPDATE]`):** Encoder replacement core logic:
```python
self.encoder = DINOv3Encoder(
    d_model=configs.d_model,
    depth=getattr(configs, "dinov3_layers", configs.e_layers),
    embed_dim=getattr(configs, "dinov3_dim", 384),
    num_heads=getattr(configs, "dinov3_heads", 6),
    mlp_ratio=getattr(configs, "dinov3_mlp_ratio", 4.0),
    n_register_tokens=getattr(configs, "dinov3_registers", 4),
    layerscale_init=getattr(configs, "dinov3_layerscale", 1e-5),
    dropout=configs.dropout,
    pretrained=getattr(configs, "dinov3_pretrained", False),
)
```
3. **`Model.forecast` (Lines 110–114):** Forward call `enc_out, attns = self.encoder(...)`. Output tensor shape is preserved and fed directly into `self.projection` (Line 98).

---

## 3. Rationale & Architecture Design

The SOFTS encoder receives multivariate series tokens after `DataEmbedding_inverted`: `[B, n_vars (+ covariates), d_model]`, rather than raw RGB pixels. DINOv3 is natively a Vision Transformer processing 2D images `B × 3 × H × W`.

The adapter mechanism implemented in `DINOv3Encoder.forward`:
1. `in_proj`: Linear adaptation from `d_model → 384` (DINOv3 width).
2. Arranges multivariate series tokens onto a near-square spatial grid, padded to $H \times W$ (padded tokens are masked with $-\infty$ during self-attention).
3. Prepends **1 [CLS] token + 4 register tokens** strictly following the official DINOv3 specification.
4. Passes tokens through pre-norm transformer blocks: Multi-head Attention + 2D Axial RoPE (applied only to patch tokens, not prefix tokens) + MLP (expansion ratio 4.0) + LayerScale ($10^{-5}$).
5. Final `LayerNorm`, prefixes and padding tokens stripped, followed by `out_proj`: `384 → d_model`.

*Note:* We avoid rasterizing numerical time series into synthetic 2D images with 16x16 patch extraction. Doing so would discard inverted temporal embedding and break the standard `[B, pred_len, C]` training interface used across time-series benchmark models.

---

## 4. Integration into Upstream SOFTS Repository

From your cloned `SOFTS/` root:

```bash
cp layers/DINOv3Encoder.py <SOFTS_ROOT>/layers/DINOv3Encoder.py
cp models/SOFTS.py <SOFTS_ROOT>/models/SOFTS.py
```

No changes to `run.py` are required. Standard training scripts remain fully functional:

```bash
python run.py \
  --model SOFTS \
  --e_layers 12 \
  --d_model 256 \
  --d_core 64 \
  --seq_len 96 --pred_len 96
```

### Pretrained Weights (Optional)

By default, the model initializes randomly according to the DINOv3 specification, allowing local forecast training without downloading ~80MB checkpoints. To load Meta AI's official weights:

```python
configs.dinov3_pretrained = True
# configs.dinov3_model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
```

Requires `pip install -U transformers` and `embed_dim=384`, `num_heads=6`.

Configuration parameters:

| Config Parameter | Default | Description |
|---|---|---|
| `dinov3_layers` | `e_layers` | Number of transformer blocks |
| `dinov3_dim` | 384 | DINOv3 hidden dimension |
| `dinov3_heads` | 6 | Number of attention heads |
| `dinov3_mlp_ratio` | 4.0 | MLP expansion ratio |
| `dinov3_registers` | 4 | Register / storage token count |
| `dinov3_layerscale` | 1e-5 | LayerScale initialization |
| `dinov3_pretrained` | False | Enable Hugging Face checkpoint loading |
| `dinov3_model_id` | `facebook/dinov3-vits16-pretrain-lvd1689m` | Hugging Face model repository ID |

---

## 5. Local Validation & Smoke Test Status

Local forward and backward tensor integrity is verified via `check_forward.py`:

```bash
pip install torch
python check_forward.py
```

| Verification Criteria | Result | Notes |
|---|---|---|
| **Forecast Tensor Shape** | `(4, 24, 7)` | Exactly matches `[B, pred_len, C]` |
| **Time Mark Support** | **Passed** | Operates with and without temporal marks |
| **Numerical Finiteness** | `torch.isfinite(y).all()` | No NaN / Inf in forward pass |
| **Encoder Parameters** | `3,602,752` | 2 ViT-S/16 blocks with 6 attention heads |

---

## 6. References

- **SOFTS Paper:** Lu et al., *SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion*, NeurIPS 2024. [arXiv:2404.14197](https://arxiv.org/abs/2404.14197).
- **DINOv3:** Meta AI Research, *DINOv3: Advancing Self-Supervised Vision Models*. [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3).
- **QCVN 05:2023/BTNMT:** National Technical Regulation on Ambient Air Quality, Ministry of Natural Resources and Environment, Vietnam.

---
