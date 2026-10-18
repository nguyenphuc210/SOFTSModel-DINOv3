# SOFTS — thay encoder bằng DINOv3

Yêu cầu 1 của bài thực hành: **thay encoder của SOFTS bằng DINOv3** (mô hình có encoder, nên không đụng backbone theo nghĩa khác). Decoder (`projection`) và embedding inverted giữ nguyên.

Repo gốc: [Secilia-Cxy/SOFTS](https://github.com/Secilia-Cxy/SOFTS) — *SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion* (NeurIPS 2024).

DINOv3: Meta, ViT-S/16 (`dinov3_vits16`, embed 384, 12 block, 6 head, 4 register token, RoPE 2D, LayerScale 1e-5). Đặc tả lấy từ [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) `dinov3/hub/backbones.py`.

---

## 1. Phạm vi thay đổi (Scope)

| Thành phần | Triển khai thực tế | Trạng thái |
|---|---|---|
| **Backbone / Encoder** | `DINOv3Encoder` trong [`layers/DINOv3Encoder.py`](layers/DINOv3Encoder.py) | **Thay thế hoàn toàn (Mới)** |
| **Replaced module** | Module `STAR` (Aggregate - Redistribute) + `EncoderLayer` | **Bỏ khỏi luồng suy luận** |
| **Preserved modules** | `DataEmbedding_inverted`, `projection` head, instance norm/denorm | **Giữ nguyên 100% chuẩn SOFTS** |

```text
Original SOFTS (NeurIPS 2024):
  x [B, T, C] → DataEmbedding_inverted → STAR Modules (Core Fusion) → Linear Projection → Forecast [B, H, C]

SOFTS-DINOv3 (Kiến trúc đề xuất):
  x [B, T, C] → DataEmbedding_inverted → DINOv3 ViT-S/16 Backbone → Linear Projection → Forecast [B, H, C]
                                           ├── Adapter in_proj (d_model → 384)
                                           ├── 2D Axial RoPE Grid + 1 CLS + 4 Register Tokens
                                           ├── 12 DINOv3 Pre-norm Blocks + LayerScale
                                           └── Adapter out_proj (384 → d_model)
```

---

## 2. Chi tiết các file thay đổi (Chỉ chỗ update cho Thầy)

Chỉ 2 file thay đổi so với bản gốc của upstream. Mọi đoạn thêm/sửa đều có comment rõ ràng `[DINOv3-UPDATE]`.

| Priority | File | Việc thực hiện | Dòng cần chỉ cho thầy |
|---|---|---|---|
| **Core (Mới)** | [`layers/DINOv3Encoder.py`](layers/DINOv3Encoder.py) | **File mới.** Triển khai DINOv3 ViT-S/16, chữ ký tương thích hoàn toàn `(x, attn_mask) -> (x, attns)`, với `x` là `[B, L, d_model]`. | Cả file |
| **Core (Sửa)** | [`models/SOFTS.py`](models/SOFTS.py) | **Sửa.** Đổi import, khởi tạo `self.encoder = DINOv3Encoder(...)` thay cho `Encoder` + `EncoderLayer` + `STAR`. | Xem 3 mốc chi tiết bên dưới |
| Base | `layers/Embed.py` | Không sửa. Copy nguyên bản để chạy độc lập. | — |
| Deprecated | `layers/Transformer_EncDec.py` | **Không còn được import.** File gốc giữ nguyên để đối chiếu. | — |

### Ba mốc sửa trong `models/SOFTS.py`:
Số dòng dưới đây là của bản đã cập nhật trong thư mục này:

1. **Import (Dòng 6–7):** Bỏ `from layers.Transformer_EncDec import Encoder, EncoderLayer`. Thêm `from layers.DINOv3Encoder import DINOv3Encoder`.
2. **`Model.__init__` (Dòng 62–95, comment `[DINOv3-UPDATE]`):** Đây là trọng tâm thay thế encoder:
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
3. **`Model.forecast` (Dòng 110–114):** Gọi forward `enc_out, attns = self.encoder(...)`. Giữ nguyên đầu ra để nối thẳng vào `self.projection` (Dòng 98).

---


## Vì sao làm kiểu này

Encoder SOFTS nhận token chuỗi sau `DataEmbedding_inverted`: `[B, n_biến (+ covariate), d_model]`, không phải ảnh RGB. DINOv3 gốc là ViT nhận ảnh `B×3×H×W`.

Cách thay, nằm trọn trong `DINOv3Encoder.forward`:

1. `in_proj`: `d_model → 384` (adapter; trọng số pretrained **không** gồm lớp này).
2. Xếp các series token thành lưới gần vuông, pad cho đủ `H×W` (pad không được attend, mask `-inf`).
3. Ghép **1 CLS + 4 register token** phía trước, đúng DINOv3.
4. Chạy các block pre-norm: Attention + RoPE 2D (chỉ trên patch token) + MLP ratio 4 + LayerScale.
5. `LayerNorm`, bỏ prefix và pad, `out_proj`: `384 → d_model`.

Không rasterize chuỗi thành ảnh rồi chạy patch-embed 16×16: làm vậy sẽ bỏ embedding inverted và phá interface `[B, pred_len, C]` mà `run.py` của SOFTS đang train. Yêu cầu là thay **encoder**, không thay cả mô hình.

`mask_k_bias` của bản pretrain DINOv3 chỉ tác dụng khi có patch bị mask (iBOT). Forecasting không mask patch, nên Q/K/V là `nn.Linear` thường — cùng đường với forward không mask.

---

## Gắn vào repo SOFTS gốc

Từ thư mục clone `SOFTS/`:

```bash
cp /đường/dẫn/softs-dinov3/layers/DINOv3Encoder.py layers/DINOv3Encoder.py
cp /đường/dẫn/softs-dinov3/models/SOFTS.py models/SOFTS.py
```

Không cần sửa `run.py`. Lệnh train cũ vẫn chạy, ví dụ:

```bash
python run.py \
  --model SOFTS \
  --e_layers 12 \
  --d_model 256 \
  --d_core 64 \
  --seq_len 96 --pred_len 96
```

`d_core` vẫn có trên argparse nhưng encoder mới không đọc. Muốn đúng độ sâu ViT-S/16 thì `--e_layers 12`.

### Trọng số pretrained (tùy chọn)

Mặc định **khởi tạo ngẫu nhiên** theo kiến trúc DINOv3, để train forecast không cần tải ~80MB. Muốn nạp checkpoint Meta, trước khi tạo model:

```python
configs.dinov3_pretrained = True
# configs.dinov3_model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"  # mặc định
```

Cần `pip install -U transformers` (bản đã có `DINOv3ViTModel`) và `embed_dim=384`, `num_heads=6`. Hàm `DINOv3Encoder.load_pretrained` copy CLS, register, từng block (`norm`, `q/k/v/o_proj`, `mlp`, LayerScale) và `norm` cuối. **Không** copy patch-embed ảnh. Nếu `e_layers < 12`, chỉ copy bấy nhiêu block đầu và in cảnh báo.

Các khóa tùy chọn khác (đều có default đúng ViT-S/16):

| configs | default | ý nghĩa |
| --- | --- | --- |
| `dinov3_layers` | `e_layers` | số block |
| `dinov3_dim` | 384 | hidden size DINOv3 |
| `dinov3_heads` | 6 | số head |
| `dinov3_mlp_ratio` | 4.0 | FFN (ViT-S; ViT-S+ là SwiGLU ratio 6, không dùng ở đây) |
| `dinov3_registers` | 4 | register / storage token |
| `dinov3_layerscale` | 1e-5 | LayerScale init |
| `dinov3_pretrained` | False | tải checkpoint HF |
| `dinov3_model_id` | `facebook/dinov3-vits16-pretrain-lvd1689m` | id checkpoint |

---

## Kiểm tra shape

```bash
pip install torch
cd softs-dinov3
python check_forward.py
```

Kỳ vọng: input `[4, 96, 7]` → output `[4, 24, 7]` cả khi có và không có time mark, giá trị hữu hạn. In ra `encoder type: DINOv3Encoder`.

---

## Luồng tensor

```text
x_enc [B, T, C]
  -> (optional) instance norm
  -> DataEmbedding_inverted        [B, C(+mark), d_model]     # không đổi
  -> DINOv3Encoder
        in_proj                    [B, L, 384]
        pad thành lưới H×W + CLS + 4 register
        × depth block (RoPE, LayerScale, MLP)
        bỏ prefix/pad, out_proj    [B, L, d_model]
  -> Linear(d_model, pred_len)     [B, L, pred_len]
  -> permute, cắt còn C biến       [B, pred_len, C]
  -> (optional) de-norm
```

---

## 5. Kết quả kiểm thử cục bộ (Validation Status / Smoke Test)

Đã kiểm tra forward pass thành công với script `check_forward.py`:

| Tiêu chí kiểm thử | Kết quả | Ghi chú |
|---|---|---|
| **Kích thước Tensor dự báo** | `(4, 24, 7)` | Khớp chính xác `[B, pred_len, C]` |
| **Hỗ trợ Time Mark** | **Passed** | Xử lý đầy đủ cả khi có/không có dấu thời gian |
| **Kiểm tra tính hữu hạn** | `torch.isfinite(y).all()` | Không bị NaN / Inf trong suy luận |
| **Tổng tham số Encoder DINOv3** | `3,602,752` | 2 blocks ViT-S/16 với 6 attention heads |

---

## 6. Tài liệu tham khảo (References)

- **SOFTS Paper:** Lu et al., *SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion*, NeurIPS 2024. [arXiv:2404.14197](https://arxiv.org/abs/2404.14197).
- **DINOv3:** Meta AI Research, *DINOv3: Advancing Self-Supervised Vision Models*. [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3).
- **QCVN 05:2023/BTNMT:** Quy chuẩn kỹ thuật quốc gia về chất lượng không khí xung quanh, Bộ Tài nguyên và Môi trường Việt Nam.

---

## 7. Hướng dẫn nộp bài lên GitHub


Để nộp link GitHub theo yêu cầu 1 của bài thực hành:

1. Tạo một repository mới trên GitHub (ví dụ: `SOFTS-DINOv3-HanoiAir`).
2. Mở terminal tại thư mục này và chạy các lệnh:
```bash
git init
git add .
git commit -m "feat: replace SOFTS encoder with DINOv3 ViT-S/16 backbone"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
git push -u origin main
```
3. Copy link GitHub repository (kèm file README.md này đã ghi rõ vị trí sửa) để nộp cho thầy.

