"""
Export NanoDet model to ONNX format WITH post-processing
=========================================================

This script exports a trained NanoDet model to ONNX format with decoded bboxes.
The output will be [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, score, class_id]

Usage:
    python export_nanodet_with_decode.py --config config_nanodet_barcode.yml --model path/to/checkpoint.pth --output nanodet_barcode.onnx
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add nanodet_src to Python path
script_dir = Path(__file__).parent.parent.parent.parent / "nanodet"  # Adjust path as needed
nanodet_src = script_dir / "nanodet_src"
if nanodet_src.exists() and str(nanodet_src) not in sys.path:
    sys.path.insert(0, str(nanodet_src))


class NanoDetWithDecode(nn.Module):
    """
    Wrapper that adds bbox decoding to NanoDet for ONNX export.
    Output format: [batch, num_boxes, 6] where 6 = [x1, y1, x2, y2, score, class_id]
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
        """Generate anchor points for all feature levels."""
        all_points = []
        all_strides = []
        
        for stride in self.strides:
            grid_size = self.input_size // stride
            # Create grid
            y, x = torch.meshgrid(
                torch.arange(grid_size, dtype=torch.float32),
                torch.arange(grid_size, dtype=torch.float32),
                indexing='ij'
            )
            # Center points
            points = torch.stack([
                x.flatten() * stride + stride / 2,
                y.flatten() * stride + stride / 2
            ], dim=1)
            all_points.append(points)
            all_strides.append(torch.full((points.shape[0],), stride, dtype=torch.float32))
        
        self.register_buffer('anchor_points', torch.cat(all_points, dim=0))
        self.register_buffer('anchor_strides', torch.cat(all_strides, dim=0))
        
    def _decode_gfl(self, predictions):
        """
        Decode GFL predictions to bboxes.
        
        predictions: [batch, num_anchors, 32 + num_classes]
        Returns: [batch, num_anchors, 6] = [x1, y1, x2, y2, score, class_id]
        """
        batch_size = predictions.shape[0]
        num_anchors = predictions.shape[1]
        
        # Split into regression and classification
        reg_preds = predictions[:, :, :32]  # [batch, num_anchors, 32]
        cls_preds = predictions[:, :, 32:]  # [batch, num_anchors, num_classes]
        
        # Decode classification (sigmoid)
        cls_scores = torch.sigmoid(cls_preds)  # [batch, num_anchors, num_classes]
        
        # Get max class score and class id
        max_scores, class_ids = cls_scores.max(dim=2)  # [batch, num_anchors]
        
        # Decode regression using DFL (Distribution Focal Loss)
        # Reshape to [batch, num_anchors, 4, reg_max+1]
        reg_preds = reg_preds.view(batch_size, num_anchors, 4, self.reg_max + 1)
        
        # Apply softmax over the distribution
        reg_dist = F.softmax(reg_preds, dim=3)  # [batch, num_anchors, 4, 8]
        
        # Weighted sum to get distance
        proj = torch.arange(self.reg_max + 1, dtype=torch.float32, device=predictions.device)
        distances = (reg_dist * proj).sum(dim=3)  # [batch, num_anchors, 4]
        
        # distances: [left, top, right, bottom]
        # Convert to bbox: [x1, y1, x2, y2]
        anchor_points = self.anchor_points.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_anchors, 2]
        anchor_strides = self.anchor_strides.unsqueeze(0).expand(batch_size, -1)  # [batch, num_anchors]
        
        # Scale distances by stride
        distances = distances * anchor_strides.unsqueeze(2)  # [batch, num_anchors, 4]
        
        x1 = anchor_points[:, :, 0] - distances[:, :, 0]
        y1 = anchor_points[:, :, 1] - distances[:, :, 1]
        x2 = anchor_points[:, :, 0] + distances[:, :, 2]
        y2 = anchor_points[:, :, 1] + distances[:, :, 3]
        
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
        # Run the original model forward
        raw_output = self.model(x)
        
        # raw_output should be [batch, num_anchors, 32 + num_classes]
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
        return False
    
    print(f"Loading config from {config_path}")
    load_config(cfg, config_path)
    
    print(f"Building model...")
    model = build_model(cfg.model)
    
    print(f"Loading weights from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Get model parameters from config
    num_classes = len(cfg.class_names) if hasattr(cfg, 'class_names') else 2
    reg_max = cfg.model.arch.head.reg_max if hasattr(cfg.model.arch.head, 'reg_max') else 7
    strides = cfg.model.arch.head.strides if hasattr(cfg.model.arch.head, 'strides') else [8, 16, 32, 64]
    
    print(f"Model config: num_classes={num_classes}, reg_max={reg_max}, strides={strides}")
    
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
    print("Testing forward pass...")
    with torch.no_grad():
        output = wrapped_model(dummy_input)
        print(f"Output shape: {output.shape}")  # Should be [1, num_anchors, 6]
    
    print(f"Exporting to ONNX: {output_path}")
    
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
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ ONNX model exported successfully!")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Output format: [batch, {output.shape[1]}, 6] = [x1, y1, x2, y2, score, class_id]")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Export NanoDet to ONNX with decoding')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='nanodet_barcode_decoded.onnx', help='Output ONNX file')
    parser.add_argument('--input-size', type=int, default=416, help='Input size')
    
    args = parser.parse_args()
    
    export_to_onnx(args.config, args.model, args.output, args.input_size)


if __name__ == "__main__":
    main()
