"""Re-export NanoDet at 416x416 for faster WASM inference."""
import sys
sys.path.insert(0, r"C:\Users\Rex\Documents\GitHub\zxing-cpp\nanodet\nanodet_src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time

INPUT_SIZE = 640
CONFIG = r"C:\Users\Rex\Documents\GitHub\zxing-cpp\nanodet\config_nanodet_barcode.yml"
WEIGHTS = r"C:\Users\Rex\Documents\GitHub\zxing-cpp\nanodet\nanodet_model_best.pth"
OUTPUT = r"C:\Users\Rex\Documents\GitHub\zxing-cpp\nanodet_barcode_416.onnx"

# Load config
from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model

load_config(cfg, CONFIG)
num_classes = cfg.model.arch.head.num_classes
reg_max = cfg.model.arch.head.reg_max
strides = cfg.model.arch.head.strides
print("Config: num_classes=%d, reg_max=%d, strides=%s" % (num_classes, reg_max, strides))

# Build model
model = build_model(cfg.model)
print("Model built")

# Load weights
checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
if 'state_dict' in checkpoint:
    sd = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
elif 'model' in checkpoint:
    sd = checkpoint['model']
else:
    sd = checkpoint

missing, unexpected = model.load_state_dict(sd, strict=False)
print("Weights loaded (missing=%d, unexpected=%d)" % (len(missing), len(unexpected)))
model.eval()


class NanoDetWithDecode(nn.Module):
    def __init__(self, model, input_size, num_classes, reg_max, strides):
        super().__init__()
        self.model = model
        self.input_size = input_size
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self._generate_anchors()

    def _generate_anchors(self):
        pts, sts = [], []
        for stride in self.strides:
            fh = math.ceil(self.input_size / stride)
            fw = math.ceil(self.input_size / stride)
            xr = torch.arange(fw, dtype=torch.float32) * stride
            yr = torch.arange(fh, dtype=torch.float32) * stride
            y, x = torch.meshgrid(yr, xr, indexing='ij')
            pts.append(torch.stack([x.flatten(), y.flatten()], dim=1))
            sts.append(torch.full((fh * fw,), stride, dtype=torch.float32))
            print("  Stride %d: %dx%d = %d anchors" % (stride, fw, fh, fh * fw))
        self.register_buffer('anchor_points', torch.cat(pts, dim=0))
        self.register_buffer('anchor_strides', torch.cat(sts, dim=0))
        print("  Total anchors: %d" % self.anchor_points.shape[0])

    def forward(self, x):
        raw = self.model(x)
        B, N, _ = raw.shape
        cls = raw[:, :, :self.num_classes]
        reg = raw[:, :, self.num_classes:]
        max_scores, class_ids = cls.max(dim=2)
        reg = reg.view(B, N, 4, self.reg_max + 1)
        reg = F.softmax(reg, dim=3)
        proj = torch.arange(self.reg_max + 1, dtype=torch.float32, device=raw.device)
        dist = (reg * proj).sum(dim=3)
        ap = self.anchor_points.unsqueeze(0).expand(B, -1, -1)
        ast = self.anchor_strides.unsqueeze(0).expand(B, -1).unsqueeze(2)
        dist = dist * ast
        x1 = torch.clamp(ap[:, :, 0] - dist[:, :, 0], 0, self.input_size)
        y1 = torch.clamp(ap[:, :, 1] - dist[:, :, 1], 0, self.input_size)
        x2 = torch.clamp(ap[:, :, 0] + dist[:, :, 2], 0, self.input_size)
        y2 = torch.clamp(ap[:, :, 1] + dist[:, :, 3], 0, self.input_size)
        return torch.stack([x1, y1, x2, y2, max_scores, class_ids.float()], dim=2)


wrapped = NanoDetWithDecode(model, INPUT_SIZE, num_classes, reg_max, strides)
wrapped.eval()

dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
with torch.no_grad():
    out = wrapped(dummy)
print("Test output: %s, score range [%.4f, %.4f]" % (
    list(out.shape), out[0, :, 4].min().item(), out[0, :, 4].max().item()))

print("Exporting ONNX at %dx%d (legacy TorchScript exporter, opset 11)..." % (INPUT_SIZE, INPUT_SIZE))
torch.onnx.export(
    wrapped, dummy, OUTPUT,
    input_names=['input'], output_names=['detections'],
    opset_version=11,
    dynamic_axes={'input': {0: 'batch_size'}, 'detections': {0: 'batch_size'}},
    dynamo=False,  # Use legacy TorchScript exporter for opset 11 compatibility
)

import onnx
m = onnx.load(OUTPUT)
onnx.checker.check_model(m)
print("ONNX validation passed")

fsize = os.path.getsize(OUTPUT)
print("Saved: %s (%d bytes, %.2f MB)" % (OUTPUT, fsize, fsize / 1024 / 1024))
print("Output: [batch, %d, 6]" % out.shape[1])

# Verify with ORT
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession(OUTPUT, providers=['CPUExecutionProvider'])
ort_out = sess.run(None, {'input': dummy.numpy()})
print("ORT verify: shape=%s max_score=%.4f" % (ort_out[0].shape, ort_out[0][0, :, 4].max()))

# Benchmark: 640 vs 416
print("\nBenchmarking 416 vs 640...")
old_path = r"C:\Users\Rex\Documents\GitHub\zxing-cpp\nanodet_barcode_clean.onnx"
if os.path.exists(old_path):
    sess640 = ort.InferenceSession(old_path, providers=['CPUExecutionProvider'])
    d640 = np.random.randn(1, 3, 640, 640).astype(np.float32)
    d416 = np.random.randn(1, 3, 416, 416).astype(np.float32)
    
    # Warmup
    for _ in range(3):
        sess640.run(None, {'input': d640})
        sess.run(None, {'input': d416})
    
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        sess640.run(None, {'input': d640})
    t640 = (time.perf_counter() - t0) / N * 1000
    
    t0 = time.perf_counter()
    for _ in range(N):
        sess.run(None, {'input': d416})
    t416 = (time.perf_counter() - t0) / N * 1000
    
    print("  640x640: %.1f ms" % t640)
    print("  416x416: %.1f ms" % t416)
    print("  Speedup: %.2fx" % (t640 / t416))

print("\nDone!")
