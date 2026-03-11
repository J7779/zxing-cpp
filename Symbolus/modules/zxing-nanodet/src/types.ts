// SPDX-License-Identifier: Apache-2.0
// ZXing + NanoDet frame processor plugin — shared TypeScript types

export interface BarcodeDetection {
  /** ZXing barcode format string, e.g. "QR_CODE", "CODE_128", "OCR", or "UNKNOWN" */
  format: string;
  /** Decoded barcode text, or OCR-extracted text when ZXing fails */
  text: string;
  /** NanoDet confidence score [0, 1] */
  confidence: number;
  /**
   * True when ZXing could not decode the barcode and PP-OCRv5 was used
   * to read the text from the detected region instead.
   */
  isOcrFallback?: boolean;
  /** Bounding box in frame pixel coordinates */
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  /** Corner points of the barcode (4 points from ZXing position) */
  cornerPoints?: Array<{ x: number; y: number }>;
  /**
   * Base64-encoded JPEG of the RAW (unrotated) luma crop from the camera sensor.
   * Only populated when `DetectBarcodesOptions.debug` is true (dev mode).
   */
  debugCropBase64?: string;
  /**
   * Detailed per-frame diagnostic log lines from the native pipeline.
   * Includes: frame dims, NanoDet detections, crop coords, luma stats,
   * rotation info, ZXing results and errors.
   * Only populated when `DetectBarcodesOptions.debug` is true (dev mode).
   */
  debugLogs?: string[];
}

export interface DetectBarcodesOptions {
  /** Minimum NanoDet confidence threshold (default: 0.35) */
  confidence?: number;
  /** NanoDet input resolution — must match the bundled ONNX model (default: 640) */
  modelInputSize?: number;
  /** Maximum detections per frame (default: 10) */
  maxDetections?: number;
  /**
   * When true, each detection includes a `debugCropBase64` JPEG of the region
   * passed to ZXing. Intended for development only — adds encoding overhead.
   */
  debug?: boolean;
}

export interface ZXingNanoDetFrameProcessorPlugin {
  detectBarcodes(
    frame: unknown,
    options?: DetectBarcodesOptions,
  ): BarcodeDetection[];
}
