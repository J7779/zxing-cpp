# NanoDet-m Barcode Detection

This folder contains scripts for training and running NanoDet-m model for barcode detection.

## About NanoDet

NanoDet is a super-fast and lightweight anchor-free object detection model. NanoDet-Plus is an enhanced version with improved accuracy while maintaining speed.

- **GitHub**: https://github.com/RangiLyu/nanodet
- **Paper**: FCOS-style anchor-free detection

### Key Features
- ⚡ Super fast: 97+ FPS on mobile CPU
- 📦 Lightweight: ~1.8M parameters for NanoDet-m
- 🎯 Anchor-free: FCOS-style detection head
- 🔧 Easy to train: Uses PyTorch Lightning

## Setup

### 1. Install Dependencies

```bash
cd nanodet
pip install -r requirements.txt
```

Or install NanoDet directly:

```bash
pip install nanodet
```

### 2. Verify Dataset

The script will automatically use the `augmented_barcodes` dataset in COCO format:

```
augmented_barcodes/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

Dataset Statistics:
- **Classes**: 1 (barcode)
- **Training images**: 28,692
- **Training annotations**: 34,608

## Training

### Quick Start

```bash
# Start training
python nanodet/nanodet_src/tools/train.py nanodet/config_nanodet_barcode.yml

# View TensorBoard logs
cd saved_models/nanodet_barcode
tensorboard --logdir ./

# Run inference after training
python nanodet/inference_nanodet.py --model saved_models/nanodet_barcode/model_best/nanodet_model_best.pth --camera 0

# Export to ONNX
python.exe export_onnx_fixed.py --config config_nanodet_barcode.yml --model ../saved_models/nanodet_barcode/model_best/nanodet_model_best.pth --output ../nanodet_barcode_fixed.onnx --input-size 640                                   
```

### With Options

```bash
# Verify dataset only
python train_nanodet.py --verify-only

# Use NanoDet CLI (recommended)
python train_nanodet.py --use-cli

# Resume from checkpoint
python train_nanodet.py --resume path/to/checkpoint.pth

# Custom config
python train_nanodet.py --config my_config.yml
```

### Configuration

The default config file `config_nanodet_barcode.yml` is set up for:
- **Model**: NanoDet-Plus with ShuffleNetV2 backbone
- **Input size**: 416x416
- **Batch size**: 32
- **Epochs**: 100
- **Optimizer**: AdamW with cosine annealing LR

Modify the config file to adjust:
- `num_classes`: Number of classes (1 for barcode)
- `total_epochs`: Training epochs
- `batchsize_per_gpu`: Batch size
- `lr`: Learning rate
- `input_size`: Model input resolution

## Inference

### Single Image

```bash
python inference_nanodet.py --model saved_models/nanodet_barcode/model_last.ckpt --image test.jpg
```

### Video

```bash
python inference_nanodet.py --model saved_models/nanodet_barcode/model_last.ckpt --video test.mp4
```

### Webcam

```bash
python inference_nanodet.py --model saved_models/nanodet_barcode/model_last.ckpt --camera 0
```

### Options

```bash
python inference_nanodet.py \
    --config config_nanodet_barcode.yml \
    --model path/to/model.pth \
    --image input.jpg \
    --output output.jpg \
    --conf-thresh 0.35
```

## Export to ONNX

```bash
python export_onnx.py \
    --config config_nanodet_barcode.yml \
    --model saved_models/nanodet_barcode/model_last.ckpt \
    --output nanodet_barcode.onnx
```

## Model Comparison

| Model | Size | Parameters | Speed (CPU) | mAP |
|-------|------|------------|-------------|-----|
| NanoDet-m | ~2MB | 0.95M | ~97 FPS | ~20 |
| NanoDet-Plus-m | ~4MB | 1.17M | ~90 FPS | ~27 |
| RF-DETR-nano | ~16MB | 3.6M | ~15 FPS | ~35 |

## Files

```
nanodet/
├── config_nanodet_barcode.yml  # Training configuration
├── train_nanodet.py            # Training script
├── inference_nanodet.py        # Inference script
├── export_onnx.py              # ONNX export script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Troubleshooting

### CUDA Out of Memory
Reduce `batchsize_per_gpu` in config file.

### Slow Training
- Reduce `workers_per_gpu` if CPU bottleneck
- Enable mixed precision: set `precision: 16`

### Low mAP
- Increase training epochs
- Adjust learning rate
- Add more data augmentation

## References

- [NanoDet GitHub](https://github.com/RangiLyu/nanodet)
- [NanoDet-Plus Paper](https://arxiv.org/abs/2111.10082)
- [COCO Format](https://cocodataset.org/#format-data)


# Quantize the nanodet barcode model (recommended)
python quantize_onnx_int8.py -i nanodet_barcode.onnx -m dynamic

# Quantize with custom output path
python quantize_onnx_int8.py -i nanodet_barcode.onnx -o nanodet_barcode_int8.onnx -m dynamic

# With verification
python quantize_onnx_int8.py -i nanodet_barcode.onnx -m dynamic --verify

# Static quantization (advanced)
python quantize_onnx_int8.py -i nanodet_barcode.onnx -m static -c path/to/calibration/images/*.jpg