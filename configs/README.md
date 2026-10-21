# Configuration Files Guide

Folder này chứa các file cấu hình YAML để training và evaluation các mô hình ViT mà không cần truyền arguments qua command line.

## Các File Cấu Hình

### 1. `train_config.yaml`
File cấu hình chính cho việc training StandardViT/LAViT model với các tham số mặc định.

### 2. `eval_config.yaml`
File cấu hình cho việc evaluation model đã được train.

## Cách Sử dụng

### Training
```bash
# Sử dụng config mặc định (StandardViT)
python src/train.py

# Sử dụng config tùy chỉnh
python src/train.py --config configs/train_config.yaml

# Hoặc tạo config file riêng của bạn
python src/train.py --config my_custom_config.yaml
```

### Evaluation
```bash
# Sử dụng config mặc định
python src/evaluate.py --checkpoint path/to/checkpoint.pth

# Sử dụng config tùy chỉnh
python src/evaluate.py --config configs/eval_config.yaml --checkpoint path/to/checkpoint.pth

# Hoặc set checkpoint trong config file và chỉ cần:
python src/evaluate.py --config my_eval_config.yaml
```

## Tùy Chỉnh Config

### Cấu Trúc File Training Config
```yaml
model:
  model_type: "StandardViT"  # hoặc "LAViT"
  img_size: 32
  num_classes: 10
  patch_size: 4
  embed_dim: 128
  depth: 6
  num_heads: 8
  # ... các tham số khác

training:
  epochs: 100
  batch_size: 128
  learning_rate: 1e-3
  # ... các tham số khác

data:
  data_dir: "./data"
  val_split: 0.1
  # ... các tham số khác

experiment:
  experiment_dir: "experiments"
  run_name: null  # auto-generated nếu null
  # ... các tham số khác
```

### Cấu Trúc File Evaluation Config
```yaml
model:
  checkpoint: "path/to/checkpoint.pth"  # bắt buộc
  config: null  # auto-detect nếu null

data:
  data_dir: "./data"
  batch_size: 128
  # ... các tham số khác

output:
  output_dir: null  # sử dụng checkpoint directory nếu null
  save_plots: true
  save_predictions: false

device:
  device: "auto"
```