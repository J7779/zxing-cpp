#!/usr/bin/env python3
"""
test_ocr.py — PP-OCRv5 mobile ONNX offline test script

Usage:
    python test_ocr.py                        # webcam (press q to quit, s to save crop)
    python test_ocr.py --image path/to/img    # single image file
    python test_ocr.py --image img --crop x,y,w,h   # crop a region first
    python test_ocr.py --text-only            # skip det, run rec on whole image

Requirements:
    pip install onnxruntime opencv-python numpy

Models are loaded from:
    Symbolus/modules/zxing-nanodet/android/src/main/assets/
"""

import argparse
import os
import sys
import time
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR   = os.path.join(SCRIPT_DIR, "Symbolus", "modules", "zxing-nanodet",
                             "android", "src", "main", "assets")
DET_MODEL    = os.path.join(ASSETS_DIR, "ppocr_v5_mobile_det.onnx")
REC_MODEL    = os.path.join(ASSETS_DIR, "ppocr_v5_mobile_rec.onnx")
DICT_FILE    = os.path.join(ASSETS_DIR, "ppocr_v5_dict.txt")

# ── load models ────────────────────────────────────────────────────────────────
try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime not found.  Run:  pip install onnxruntime opencv-python numpy")

try:
    import cv2
except ImportError:
    sys.exit("opencv-python not found.  Run:  pip install opencv-python")

print(f"[INFO] onnxruntime {ort.__version__}")
print(f"[INFO] loading det:  {DET_MODEL}")
print(f"[INFO] loading rec:  {REC_MODEL}")

for p in [DET_MODEL, REC_MODEL, DICT_FILE]:
    if not os.path.exists(p):
        sys.exit(f"[ERROR] missing file: {p}")

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

sess_det = ort.InferenceSession(DET_MODEL, sess_options=so)
sess_rec = ort.InferenceSession(REC_MODEL, sess_options=so)

det_in    = sess_det.get_inputs()[0]
det_out   = sess_det.get_outputs()[0]
rec_in    = sess_rec.get_inputs()[0]
rec_out   = sess_rec.get_outputs()[0]

print(f"[INFO] det  in={det_in.name} {det_in.shape}  out={det_out.name} {det_out.shape}")
print(f"[INFO] rec  in={rec_in.name} {rec_in.shape}  out={rec_out.name} {rec_out.shape}")

with open(DICT_FILE, "r", encoding="utf-8") as f:
    chars = f.read().splitlines()   # chars[i] → class index i+1
# Class 0 = CTC blank;  1..len(chars) = chars[idx-1];  len(chars)+1 = space
DICT = chars
SPACE_IDX = len(chars) + 1
print(f"[INFO] dict size={len(DICT)}  space_idx={SPACE_IDX}  total_classes={len(DICT)+2}")


# ── preprocessing ──────────────────────────────────────────────────────────────

def preprocess_det(bgr: np.ndarray, max_dim: int = 960) -> tuple[np.ndarray, float, int, int]:
    """Resize to max_dim with 32-aligned sides, ImageNet normalize, return (tensor, scale, w, h)."""
    h, w = bgr.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    new_w = max(32, int(w * scale) // 32 * 32)
    new_h = max(32, int(h * scale) // 32 * 32)
    resized = cv2.resize(bgr, (new_w, new_h))
    img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # BGR → RGB, HWC → CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    return img[np.newaxis].astype(np.float32), scale, new_w, new_h


def preprocess_rec(bgr: np.ndarray, target_h: int = 48, max_w: int = 1200) -> np.ndarray:
    """Resize to target_h keeping aspect ratio, normalize (v-0.5)/0.5, return NCHW tensor."""
    h, w = bgr.shape[:2]
    new_w = max(8, min(max_w, int(w / h * target_h)))
    resized = cv2.resize(bgr, (new_w, target_h)).astype(np.float32) / 255.0
    img = (resized - 0.5) / 0.5
    # BGR → RGB, HWC → CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    return img[np.newaxis].astype(np.float32)


# ── CTC decode ─────────────────────────────────────────────────────────────────

def ctc_greedy(logits: np.ndarray) -> str:
    """logits shape: (1, seq_len, num_classes). Returns decoded string."""
    tokens = np.argmax(logits[0], axis=-1)   # (seq_len,)
    text = []
    prev = -1
    for tok in tokens:
        if tok != 0 and tok != prev:          # 0 = blank
            if tok == SPACE_IDX:
                text.append(" ")
            elif 1 <= tok <= len(DICT):
                text.append(DICT[tok - 1])
            # else: out-of-range, skip
        prev = tok
    return "".join(text).strip()


# ── detection post-processing ──────────────────────────────────────────────────

def db_postprocess(prob_map: np.ndarray, orig_h: int, orig_w: int,
                   scale: float, threshold: float = 0.3,
                   min_size: int = 5) -> list[tuple[int, int, int, int]]:
    """
    Simple DB post-processing: threshold the 1-channel probability map, find
    connected components, return bboxes in original image coordinates.
    """
    prob = prob_map[0, 0]          # (H, W)
    binary = (prob > threshold).astype(np.uint8) * 255
    # Dilate slightly to merge nearby text
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    dh, dw = prob.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_size or h < min_size:
            continue
        # Map back to original image coordinates
        ox = int(x / scale)
        oy = int(y / scale)
        ow = int(w / scale)
        oh = int(h / scale)
        ox, oy = max(0, ox), max(0, oy)
        ow = min(ow, orig_w - ox)
        oh = min(oh, orig_h - oy)
        if ow > 0 and oh > 0:
            boxes.append((ox, oy, ow, oh))
    return boxes


# ── full OCR pipeline ──────────────────────────────────────────────────────────

def run_det(bgr: np.ndarray, threshold: float = 0.3) -> list[tuple[int, int, int, int]]:
    """Run DB text detection. Returns list of (x, y, w, h) boxes in original coords."""
    tensor, scale, nw, nh = preprocess_det(bgr)
    t0 = time.perf_counter()
    out = sess_det.run([det_out.name], {det_in.name: tensor})[0]
    dt = time.perf_counter() - t0
    print(f"  [DET] input={nw}x{nh} scale={scale:.3f} output={out.shape}  ({dt*1000:.1f}ms)")
    return db_postprocess(out, bgr.shape[0], bgr.shape[1], scale, threshold)


def run_rec(bgr: np.ndarray, label: str = "") -> str:
    """Run CRNN text recognition on a BGR image strip."""
    tensor = preprocess_rec(bgr)
    orig_h, orig_w = bgr.shape[:2]
    t0 = time.perf_counter()
    logits = sess_rec.run([rec_out.name], {rec_in.name: tensor})[0]
    dt = time.perf_counter() - t0
    text = ctc_greedy(logits)
    seq_len = logits.shape[1]
    print(f"  [REC{label}] input={tensor.shape[3]}x{tensor.shape[2]} "
          f"(orig {orig_w}x{orig_h}) seqLen={seq_len} → '{text}'  ({dt*1000:.1f}ms)")
    return text


def ocr_image(bgr: np.ndarray, det_threshold: float = 0.3,
              skip_det: bool = False) -> list[dict]:
    """
    Full OCR pipeline on a BGR image.
    Returns list of {'box': (x,y,w,h), 'text': str}.
    """
    results = []
    if skip_det:
        text = run_rec(bgr, " WHOLE")
        results.append({"box": (0, 0, bgr.shape[1], bgr.shape[0]), "text": text})
        return results

    boxes = run_det(bgr, det_threshold)
    print(f"  [DET] found {len(boxes)} text region(s)")
    if not boxes:
        print("  [DET] no text found — falling back to whole-image recognition")
        text = run_rec(bgr, " WHOLE")
        results.append({"box": (0, 0, bgr.shape[1], bgr.shape[0]), "text": text})
        return results

    for i, (bx, by, bw, bh) in enumerate(boxes):
        crop = bgr[by:by+bh, bx:bx+bw]
        if crop.size == 0:
            continue
        text = run_rec(crop, f" #{i}")
        results.append({"box": (bx, by, bw, bh), "text": text})
    return results


def draw_results(bgr: np.ndarray, results: list[dict]) -> np.ndarray:
    vis = bgr.copy()
    for r in results:
        bx, by, bw, bh = r["box"]
        text = r["text"]
        color = (0, 220, 255) if text else (80, 80, 80)
        cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), color, 2)
        label = text or "(no text)"
        cv2.putText(vis, label, (bx, max(by - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="PP-OCRv5 ONNX test")
    ap.add_argument("--image", "-i", default=None, help="Input image path")
    ap.add_argument("--crop", "-c", default=None,
                    help="Crop before OCR: x,y,w,h in pixels")
    ap.add_argument("--det-threshold", type=float, default=0.3,
                    help="DB detection confidence threshold (default 0.3)")
    ap.add_argument("--text-only", action="store_true",
                    help="Skip detection, run recognition on entire image")
    ap.add_argument("--hri", action="store_true",
                    help="Extract only bottom 20pct of crop (barcode HRI strip mode)")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.image:
        # ── single image mode ────────────────────────────────────────────
        bgr = cv2.imread(args.image)
        if bgr is None:
            sys.exit(f"[ERROR] cannot read image: {args.image}")
        print(f"[IMAGE] {args.image}  size={bgr.shape[1]}x{bgr.shape[0]}")

        if args.crop:
            parts = list(map(int, args.crop.split(",")))
            cx, cy, cw, ch = parts[0], parts[1], parts[2], parts[3]
            bgr = bgr[cy:cy+ch, cx:cx+cw]
            print(f"[CROP]  applied ({cx},{cy} {cw}x{ch}) → {bgr.shape[1]}x{bgr.shape[0]}")

        if args.hri:
            h = bgr.shape[0]
            bgr = bgr[int(h * 0.70):, :]
            print(f"[HRI]   bottom-30pct strip → {bgr.shape[1]}x{bgr.shape[0]}")

        cv2.imshow("input", bgr)
        results = ocr_image(bgr, args.det_threshold, skip_det=args.text_only)

        print("\n── Results ──")
        for r in results:
            print(f"  box={r['box']}  text='{r['text']}'")

        vis = draw_results(bgr, results)
        cv2.imshow("OCR result", vis)
        print("\nPress any key to quit…")
        cv2.waitKey(0)

    else:
        # ── webcam / live mode ───────────────────────────────────────────
        print("[CAM] opening default camera…  (q=quit  s=save  h=HRI strip)")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("[ERROR] cannot open webcam")

        hri_mode = args.hri
        save_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                fname = f"ocr_capture_{save_idx:04d}.png"
                cv2.imwrite(fname, frame)
                print(f"[SAVE] {fname}")
                save_idx += 1
            if key == ord("h"):
                hri_mode = not hri_mode
                print(f"[HRI] mode {'ON' if hri_mode else 'OFF'}")

            img = frame.copy()
            if hri_mode:
                img = img[int(img.shape[0] * 0.70):, :]

            results = ocr_image(img, args.det_threshold, skip_det=args.text_only)
            vis = draw_results(img, results)
            status = f"HRI={'ON' if hri_mode else 'OFF'}  q=quit s=save h=toggle-hri"
            cv2.putText(vis, status, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("PP-OCRv5 live", vis)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
