"""
NanoDet Inference Script for Barcode Detection
===============================================

This script provides inference utilities for the trained NanoDet-m model.

Usage:
    python inference_nanodet.py --image path/to/image.jpg
    python inference_nanodet.py --video path/to/video.mp4
    python inference_nanodet.py --camera 0
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch
from pathlib import Path

# Add nanodet_src to Python path
script_dir = Path(__file__).parent
nanodet_src_path = script_dir / "nanodet_src"
if str(nanodet_src_path) not in sys.path:
    sys.path.insert(0, str(nanodet_src_path))


def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import torch
        import nanodet
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False


class NanoDetInference:
    """NanoDet inference wrapper for barcode detection."""
    
    def __init__(self, config_path: str, model_path: str, conf_thresh: float = 0.35):
        """
        Initialize NanoDet inference.
        
        Args:
            config_path: Path to the YAML config file
            model_path: Path to the trained model checkpoint
            conf_thresh: Confidence threshold for detections
        """
        from nanodet.util import cfg, load_config
        from nanodet.model.arch import build_model
        from nanodet.util import load_model_weight
        from nanodet.data.transform import Pipeline
        import torch
        
        self.conf_thresh = conf_thresh
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        load_config(cfg, config_path)
        self.cfg = cfg
        
        # Build and load model
        print(f"Loading model from {model_path}...")
        self.model = build_model(cfg.model)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present (from pytorch-lightning)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessing pipeline
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        self.input_size = cfg.data.val.input_size
        
        # Class names
        self.class_names = cfg.class_names if hasattr(cfg, 'class_names') else ['barcode']
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Classes: {self.class_names}")
    
    def preprocess(self, image: np.ndarray):
        """Preprocess image for inference."""
        from nanodet.data.batch_process import stack_batch_img
        from nanodet.data.collate import naive_collate
        import torch
        
        height, width = image.shape[:2]
        img_info = {'id': 0, 'file_name': None, 'height': height, 'width': width}
        
        # Apply preprocessing pipeline
        meta = dict(img_info=img_info, raw_img=image, img=image)
        meta = self.pipeline(None, meta, self.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta['img'] = stack_batch_img(meta['img'], divisible=32)
        
        return meta
    
    def postprocess(self, dets, conf_thresh=None):
        """Postprocess model predictions."""
        if conf_thresh is None:
            conf_thresh = self.conf_thresh
        
        results = []
        
        # dets is a dict: {img_id: {class_id: [[x1, y1, x2, y2, score], ...]}}
        # Get the first image's results (img_id = 0)
        if 0 not in dets:
            return results
        
        det_results = dets[0]
        
        # Iterate through each class
        for class_id, class_dets in det_results.items():
            # class_dets is a list of [x1, y1, x2, y2, score]
            for det in class_dets:
                if len(det) >= 5:
                    x1, y1, x2, y2, score = det[:5]
                    
                    if score >= conf_thresh:
                        results.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'score': float(score),
                            'class_id': int(class_id),
                            'class_name': self.class_names[int(class_id)] if int(class_id) < len(self.class_names) else f'class_{class_id}'
                        })
        
        return results
    
    @torch.no_grad()
    def detect(self, image: np.ndarray):
        """
        Run detection on an image.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of detections with bbox, score, class_id, class_name
        """
        # Preprocess
        meta = self.preprocess(image)
        
        # Inference - use model's inference method which handles post-processing
        dets = self.model.inference(meta)
        
        # Postprocess
        results = self.postprocess(dets)
        
        return results
    
    def draw_results(self, image: np.ndarray, results: list, show_score: bool = True):
        """Draw detection results on image."""
        img = image.copy()
        
        for det in results:
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            class_name = det['class_name']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_score:
                label = f"{class_name}: {score:.2f}"
            else:
                label = class_name
            
            # Label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                img, 
                (x1, y1 - label_h - 10), 
                (x1 + label_w, y1), 
                color, 
                -1
            )
            cv2.putText(
                img, label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 0, 0), 2
            )
        
        return img


def is_detection_in_roi(bbox, roi):
    """
    Check if a detection bounding box is within the ROI.
    
    Args:
        bbox: [x1, y1, x2, y2] detection bounding box
        roi: [x1, y1, x2, y2] region of interest
    
    Returns:
        True if bbox center is within ROI
    """
    if roi is None:
        return True
    
    # Calculate center of detection bbox
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2
    
    # Check if center is within ROI
    return (roi[0] <= bbox_center_x <= roi[2] and 
            roi[1] <= bbox_center_y <= roi[3])


def draw_roi(image, roi, color=(255, 0, 0), thickness=3):
    """
    Draw the region of interest on the image.
    
    Args:
        image: Image to draw on
        roi: [x1, y1, x2, y2] region of interest
        color: Color of the ROI rectangle (default: blue)
        thickness: Thickness of the ROI rectangle
    """
    if roi is not None:
        cv2.rectangle(image, (roi[0], roi[1]), (roi[2], roi[3]), color, thickness)
        # Add label
        cv2.putText(
            image, 
            'Detection Area', 
            (roi[0], roi[1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, color, 2
        )
    return image


def run_image_inference(detector, image_path: str, output_path: str = None, show: bool = True):
    """Run inference on a single image."""
    print(f"\nProcessing image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Run detection
    start_time = time.time()
    results = detector.detect(image)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time*1000:.2f}ms")
    print(f"Detections: {len(results)}")
    
    for det in results:
        print(f"  - {det['class_name']}: {det['score']:.2f} at {det['bbox']}")
    
    # Draw results
    result_image = detector.draw_results(image, results)
    
    # Save or show
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to: {output_path}")
    
    if show:
        cv2.imshow('NanoDet Barcode Detection', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video_inference(detector, video_path: str, output_path: str = None, roi_size: float = 0.6):
    """Run inference on a video file.
    
    Args:
        detector: NanoDet detector instance
        video_path: Path to video file
        output_path: Path to save output video
        roi_size: Size of detection area as fraction of frame (0.0-1.0, default 0.6)
    """
    print(f"\nProcessing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate ROI (centered square) - use smaller dimension to ensure square fits
    min_dimension = min(width, height)
    roi_size_px = int(min_dimension * roi_size)
    roi_x1 = (width - roi_size_px) // 2
    roi_y1 = (height - roi_size_px) // 2
    roi_x2 = roi_x1 + roi_size_px
    roi_y2 = roi_y1 + roi_size_px
    roi = [roi_x1, roi_y1, roi_x2, roi_y2]
    
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"Detection area (ROI): {roi_size_px}x{roi_size_px} square at center")
    
    # Setup video writer if output path specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        start_time = time.time()
        results = detector.detect(frame)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Filter results by ROI
        filtered_results = [r for r in results if is_detection_in_roi(r['bbox'], roi)]
        
        # Draw ROI
        result_frame = draw_roi(frame.copy(), roi)
        
        # Draw results
        result_frame = detector.draw_results(result_frame, filtered_results)
        
        # Add FPS text
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(
            result_frame, 
            f'FPS: {current_fps:.1f}', 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2
        )
        
        # Write or display
        if writer:
            writer.write(result_frame)
        
        cv2.imshow('NanoDet Barcode Detection', result_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            avg_fps = frame_count / total_time
            print(f"Processed {frame_count}/{total_frames} frames, Avg FPS: {avg_fps:.1f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if writer:
        writer.release()
        print(f"Result saved to: {output_path}")
    cv2.destroyAllWindows()
    
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nProcessed {frame_count} frames in {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.1f}")


def run_camera_inference(detector, camera_id: int = 0, roi_size: float = 0.6):
    """Run real-time inference on camera feed.
    
    Args:
        detector: NanoDet detector instance
        camera_id: Camera device ID
        roi_size: Size of detection area as fraction of frame (0.0-1.0, default 0.6)
    """
    print(f"\nStarting camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Query and set maximum supported resolution
    # Try common high resolutions and use the highest one that works
    resolutions = [
        (3840, 2160),  # 4K
        (2560, 1440),  # 1440p
        (1920, 1080),  # 1080p
        (1280, 720),   # 720p
        (640, 480)     # VGA (fallback)
    ]
    
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width >= width * 0.9 and actual_height >= height * 0.9:
            print(f"Camera resolution set to: {actual_width}x{actual_height}")
            break
    else:
        # If none worked, get whatever the camera defaulted to
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Using default camera resolution: {actual_width}x{actual_height}")
    
    # Calculate ROI (centered square) - use smaller dimension to ensure square fits
    min_dimension = min(actual_width, actual_height)
    roi_size_px = int(min_dimension * roi_size)
    roi_x1 = (actual_width - roi_size_px) // 2
    roi_y1 = (actual_height - roi_size_px) // 2
    roi_x2 = roi_x1 + roi_size_px
    roi_y2 = roi_y1 + roi_size_px
    roi = [roi_x1, roi_y1, roi_x2, roi_y2]
    
    print(f"Detection area (ROI): {roi_size_px}x{roi_size_px} square at center")
    print("Press 'q' to quit, '+' to increase confidence, '-' to decrease confidence")
    
    # Dynamic confidence threshold
    conf_thresh = detector.conf_thresh
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Run detection
        start_time = time.time()
        # Temporarily override confidence threshold
        original_thresh = detector.conf_thresh
        detector.conf_thresh = conf_thresh
        results = detector.detect(frame)
        detector.conf_thresh = original_thresh
        inference_time = time.time() - start_time
        total_time += inference_time
        frame_count += 1
        
        # Filter results by ROI
        filtered_results = [r for r in results if is_detection_in_roi(r['bbox'], roi)]
        
        # Draw ROI
        result_frame = draw_roi(frame.copy(), roi)
        
        # Draw results
        result_frame = detector.draw_results(result_frame, filtered_results)
        
        # Add FPS, detection count, and confidence threshold
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(
            result_frame, 
            f'FPS: {current_fps:.1f} | Detections: {len(filtered_results)}/{len(results)} | Conf: {conf_thresh:.2f}', 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, (0, 255, 0), 2
        )
        
        cv2.imshow('NanoDet Barcode Detection', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            conf_thresh = min(0.95, conf_thresh + 0.05)
            print(f"Confidence threshold: {conf_thresh:.2f}")
        elif key == ord('-') or key == ord('_'):
            conf_thresh = max(0.05, conf_thresh - 0.05)
            print(f"Confidence threshold: {conf_thresh:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nProcessed {frame_count} frames")
    print(f"Average FPS: {avg_fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description='NanoDet Barcode Detection Inference')
    parser.add_argument('--config', type=str, default='config_nanodet_barcode.yml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to input video')
    parser.add_argument('--camera', type=int, default=None,
                        help='Camera ID for real-time detection')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output')
    parser.add_argument('--conf-thresh', type=float, default=0.35,
                        help='Confidence threshold')
    parser.add_argument('--roi-size', type=float, default=0.6,
                        help='Size of detection area as fraction of frame (0.0-1.0, default 0.6)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display results')
    
    args = parser.parse_args()
    
    if not check_dependencies():
        print("\nPlease install required dependencies:")
        print("  pip install -r requirements.txt")
        return
    
    # Get config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    # Initialize detector
    detector = NanoDetInference(config_path, args.model, args.conf_thresh)
    
    # Run inference
    if args.image:
        run_image_inference(detector, args.image, args.output, not args.no_show)
    elif args.video:
        run_video_inference(detector, args.video, args.output, args.roi_size)
    elif args.camera is not None:
        run_camera_inference(detector, args.camera, args.roi_size)
    else:
        print("Please specify --image, --video, or --camera")
        parser.print_help()


if __name__ == "__main__":
    main()
