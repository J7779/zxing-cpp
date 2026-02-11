"""
Export NanoDet model to ONNX format WITH post-processing (FIXED)
================================================================

This script exports a trained NanoDet model to ONNX format with decoded bboxes.
The output will be [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, score, class_id]

FIXES:
1. Correct order of cls_preds and reg_preds (NanoDet's _forward_onnx puts cls first with sigmoid)
2. Correct anchor point generation to match NanoDet exactly
3. Proper stride handling

Usage:
    python export_onnx_fixed.py --config config_nanodet_barcode.yml --model path/to/checkpoint.pth --output nanodet_barcode.onnx
"""

import argparse
import os
import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add nanodet_src to Python path
script_dir = Path(__file__).parent
nanodet_src = script_dir / "nanodet_src"
if nanodet_src.exists() and str(nanodet_src) not in sys.path:
    sys.path.insert(0, str(nanodet_src))


class NanoDetWithDecode(nn.Module):
    """
    Wrapper that adds bbox decoding to NanoDet for ONNX export.
    Output format: [batch, num_boxes, 6] where 6 = [x1, y1, x2, y2, score, class_id]
    
    IMPORTANT: NanoDet's _forward_onnx outputs cls_preds FIRST (with sigmoid applied),
    then reg_preds. The format is [batch, num_anchors, num_classes + 4*(reg_max+1)]
    """
    
    def __init__(self, model, input_size=416, num_classes=2, reg_max=7, strides=[8, 16, 32, 64]):
        super().__init__()
        self.model = model
        self.input_size = input_size
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        
        # Pre-compute anchor points (will be registered as buffers)
        self._generate_anchor_points()
        
    def _generate_anchor_points(self):
        """
        Generate anchor points for all feature levels.
        Matches NanoDet's get_single_level_center_priors exactly.
        """
        all_points = []
        all_strides = []
        
        for stride in self.strides:
            # Use ceiling division to match NanoDet's grid generation
            feat_h = math.ceil(self.input_size / stride)
            feat_w = math.ceil(self.input_size / stride)
            
            # NanoDet uses: x_range = (torch.arange(w)) * stride
            # and: y_range = (torch.arange(h)) * stride
            # Then meshgrid with y_range first (default indexing='ij')
            x_range = torch.arange(feat_w, dtype=torch.float32) * stride
            y_range = torch.arange(feat_h, dtype=torch.float32) * stride
            
            y, x = torch.meshgrid(y_range, x_range, indexing='ij')
            
            # NanoDet stores as [x, y] after flattening
            points = torch.stack([x.flatten(), y.flatten()], dim=1)
            all_points.append(points)
            all_strides.append(torch.full((points.shape[0],), stride, dtype=torch.float32))
        
        self.register_buffer('anchor_points', torch.cat(all_points, dim=0))
        self.register_buffer('anchor_strides', torch.cat(all_strides, dim=0))
        
        print(f"Generated {self.anchor_points.shape[0]} anchor points")
        for i, stride in enumerate(self.strides):
            feat_size = math.ceil(self.input_size / stride)
            print(f"  Stride {stride}: {feat_size}x{feat_size} = {feat_size*feat_size} anchors")
        
    def _decode_gfl(self, predictions):
        """
        Decode GFL predictions to bboxes.
        
        NanoDet's _forward_onnx output format:
        - predictions: [batch, num_anchors, num_classes + 4*(reg_max+1)]
        - First num_classes channels are class scores (ALREADY SIGMOIDED)
        - Remaining 4*(reg_max+1) channels are regression distribution
        
        Returns: [batch, num_anchors, 6] = [x1, y1, x2, y2, score, class_id]
        """
        batch_size = predictions.shape[0]
        num_anchors = predictions.shape[1]
        reg_channels = 4 * (self.reg_max + 1)  # 4 * 8 = 32 for reg_max=7
        
        # Split: cls_preds FIRST, then reg_preds (this is the key fix!)
        cls_preds = predictions[:, :, :self.num_classes]  # [batch, num_anchors, num_classes]
        reg_preds = predictions[:, :, self.num_classes:]  # [batch, num_anchors, 32]
        
        # cls_preds already has sigmoid applied in _forward_onnx
        cls_scores = cls_preds
        
        # Get max class score and class id
        max_scores, class_ids = cls_scores.max(dim=2)  # [batch, num_anchors]
        
        # Decode regression using DFL (Distribution Focal Loss) / Integral
        # Reshape to [batch, num_anchors, 4, reg_max+1]
        reg_preds = reg_preds.view(batch_size, num_anchors, 4, self.reg_max + 1)
        
        # Apply softmax over the distribution dimension
        reg_dist = F.softmax(reg_preds, dim=3)  # [batch, num_anchors, 4, 8]
        
        # Weighted sum to get distance (this is the Integral operation)
        proj = torch.arange(self.reg_max + 1, dtype=torch.float32, device=predictions.device)
        distances = (reg_dist * proj).sum(dim=3)  # [batch, num_anchors, 4]
        
        # distances: [left, top, right, bottom] from center
        # Convert to bbox: [x1, y1, x2, y2]
        anchor_points = self.anchor_points.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_anchors, 2]
        anchor_strides = self.anchor_strides.unsqueeze(0).expand(batch_size, -1)  # [batch, num_anchors]
        
        # Scale distances by stride (NanoDet's dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2])
        distances = distances * anchor_strides.unsqueeze(2)  # [batch, num_anchors, 4]
        
        # distance2bbox: x1 = cx - left, y1 = cy - top, x2 = cx + right, y2 = cy + bottom
        x1 = anchor_points[:, :, 0] - distances[:, :, 0]
        y1 = anchor_points[:, :, 1] - distances[:, :, 1]
        x2 = anchor_points[:, :, 0] + distances[:, :, 2]
        y2 = anchor_points[:, :, 1] + distances[:, :, 3]
        
        # Clamp to image bounds
        x1 = torch.clamp(x1, min=0, max=self.input_size)
        y1 = torch.clamp(y1, min=0, max=self.input_size)
        x2 = torch.clamp(x2, min=0, max=self.input_size)
        y2 = torch.clamp(y2, min=0, max=self.input_size)
        
        # Stack results: [x1, y1, x2, y2, score, class_id]
        bboxes = torch.stack([
            x1, y1, x2, y2,
            max_scores,
            class_ids.float()
        ], dim=2)  # [batch, num_anchors, 6]
        
        return bboxes
    
    def forward(self, x):
        """
        Forward pass with decoding.
        
        Args:
            x: Input image tensor [batch, 3, H, W]
            
        Returns:
            Decoded detections [batch, num_anchors, 6]
            where 6 = [x1, y1, x2, y2, score, class_id]
        """
        # Run the original model forward (this calls _forward_onnx when in ONNX export mode)
        raw_output = self.model(x)
        
        # raw_output: [batch, num_anchors, num_classes + 4*(reg_max+1)]
        # Decode to bboxes
        decoded = self._decode_gfl(raw_output)
        
        return decoded


def export_to_onnx(config_path: str, model_path: str, output_path: str, input_size: int = 416):
    """
    Export NanoDet model to ONNX format with decoded outputs.
    """
    try:
        import onnx
        from nanodet.util import cfg, load_config
        from nanodet.model.arch import build_model
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you have nanodet installed or nanodet_src in the path")
        return False
    
    print(f"Loading config from {config_path}")
    load_config(cfg, config_path)
    
    print(f"Building model...")
    model = build_model(cfg.model)
    
    print(f"Loading weights from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present (from pytorch-lightning)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Get model parameters from config
    num_classes = cfg.model.arch.head.num_classes
    reg_max = cfg.model.arch.head.reg_max
    strides = cfg.model.arch.head.strides
    
    print(f"\nModel config:")
    print(f"  num_classes: {num_classes}")
    print(f"  reg_max: {reg_max}")
    print(f"  strides: {strides}")
    print(f"  input_size: {input_size}")
    
    # Wrap model with decoder
    wrapped_model = NanoDetWithDecode(
        model, 
        input_size=input_size, 
        num_classes=num_classes,
        reg_max=reg_max,
        strides=strides
    )
    wrapped_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = wrapped_model(dummy_input)
        print(f"Output shape: {output.shape}")  # Should be [1, num_anchors, 6]
        
        # Check output ranges
        print(f"\nOutput analysis (random input):")
        print(f"  Scores range: {output[0, :, 4].min():.4f} to {output[0, :, 4].max():.4f}")
        print(f"  x1 range: {output[0, :, 0].min():.1f} to {output[0, :, 0].max():.1f}")
        print(f"  y1 range: {output[0, :, 1].min():.1f} to {output[0, :, 1].max():.1f}")
        print(f"  x2 range: {output[0, :, 2].min():.1f} to {output[0, :, 2].max():.1f}")
        print(f"  y2 range: {output[0, :, 3].min():.1f} to {output[0, :, 3].max():.1f}")
    
    print(f"\nExporting to ONNX: {output_path}")
    
    # Export to ONNX
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['detections'],
        opset_version=11,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'detections': {0: 'batch_size'}
        }
    )
    
    # Verify
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Simplify if possible
    try:
        from onnxsim import simplify
        print("Simplifying ONNX model...")
        model_simplified, check = simplify(onnx_model)
        if check:
            onnx.save(model_simplified, output_path)
            print("Model simplified successfully")
    except ImportError:
        print("onnx-simplifier not installed, skipping")
    except Exception as e:
        print(f"Simplification failed: {e}")
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"✓ ONNX model exported successfully!")
    print(f"{'='*60}")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Output format: [batch, {output.shape[1]}, 6]")
    print(f"  Output meaning: [x1, y1, x2, y2, score, class_id]")
    print(f"\nPreprocessing requirements:")
    print(f"  - Input size: {input_size}x{input_size}")
    print(f"  - BGR format (not RGB!)")
    print(f"  - Normalization: (img - mean) / std")
    print(f"    mean = [103.53, 116.28, 123.675] (BGR)")
    print(f"    std  = [57.375, 57.12, 58.395] (BGR)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Export NanoDet to ONNX with decoding (FIXED)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='nanodet_barcode_decoded.onnx', help='Output ONNX file')
    parser.add_argument('--input-size', type=int, default=416, help='Input size')
    
    args = parser.parse_args()
    
    export_to_onnx(args.config, args.model, args.output, args.input_size)


if __name__ == "__main__":
    main()
