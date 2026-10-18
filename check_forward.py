"""Shape check: SOFTS with the DINOv3 encoder still forecasts [B, pred_len, C]."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.SOFTS import Model


class Configs:
    seq_len = 96
    pred_len = 24
    d_model = 64
    d_core = 32          # unused by the new encoder; kept so old scripts still parse
    d_ff = 128           # unused
    dropout = 0.1
    e_layers = 2         # DINOv3 depth (official ViT-S/16 is 12)
    activation = "gelu" # unused
    use_norm = True
    dinov3_pretrained = False


def main():
    torch.manual_seed(0)
    cfg = Configs()
    model = Model(cfg)
    model.train()
    x = torch.randn(4, cfg.seq_len, 7)
    # time mark: 4 calendar features, same layout SOFTS uses
    x_mark = torch.randn(4, cfg.seq_len, 4)
    y = model(x, x_mark, None, None)
    assert y.shape == (4, cfg.pred_len, 7), y.shape

    model.eval()
    with torch.no_grad():
        y_eval = model(x, None, None, None)
    assert y_eval.shape == (4, cfg.pred_len, 7), y_eval.shape
    assert torch.isfinite(y_eval).all()

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"ok  y={tuple(y.shape)}  params={n_params}  dinov3_encoder={n_enc}")
    print(f"encoder type: {type(model.encoder).__name__}  depth={model.encoder.depth}")


if __name__ == "__main__":
    main()
