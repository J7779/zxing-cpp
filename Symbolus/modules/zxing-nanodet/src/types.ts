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
   * Source of the detection: 'nanodet' (model-detected region), 'direct' (scan-box
   * ZXing pass), or 'consensus' (verified by consensus algorithm).
   */
  source?: 'nanodet' | 'direct' | 'consensus';
  /**
   * When enableDamagedBarcode is true, the merged text from partial ZXing +
   * OCR reads. Present only when the merge actually contributed.
   */
  mergedText?: string;
  /**
   * Number of consistent reads that confirmed this result via the consensus
   * algorithm. Only populated when enableConsensus is true.
   */
  consensusCount?: number;
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
  /**
   * Monotonically increasing ID from the native inference pipeline.
   * Used by the consensus algorithm to distinguish new inference frames
   * from cached repeats of the same results. Internal use only.
   */
  _inferenceId?: number;
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
  /**
   * When false, the ZXing structured barcode decoder is skipped entirely.
   * Useful for testing OCR in isolation. Default: true.
   */
  enableZxing?: boolean;
  /**
   * When false, the PP-OCRv5 fallback is never invoked even when ZXing fails.
   * Useful for testing ZXing in isolation. Default: false.
   */
  enableOcr?: boolean;
  /**
   * Restrict which barcode formats are reported. Exact ZXing-C++ format strings
   * (e.g. "Code128", "QRCode", "EAN-13", "ITF"). When omitted or empty, all
   * formats are accepted. Filtering happens in Kotlin after ZXing decodes.
   */
  enabledFormats?: string[];

  // ── New advanced features ─────────────────────────────────────────────

  /**
   * Run ZXing directly on the full scan-box region in parallel with NanoDet.
   * If NanoDet misses a barcode, the direct ZXing pass may still decode it.
   * Default: true.
   */
  enableDirectZxing?: boolean;

  /**
   * Resolution multiplier for the crop passed to ZXing. Values > 1 upscale
   * the region for better decode rates on small or dense barcodes.
   * Default: 1.0 (native resolution).
   */
  zxingResolutionScale?: number;

  /**
   * Torch / flash mode for low-light environments.
   *   'off'  — torch always off
   *   'on'   — torch always on
   *   'auto' — native code enables torch when frame brightness is below threshold
   * Default: 'off'.
   */
  torchMode?: 'off' | 'on' | 'auto';

  /**
   * Target luma threshold (0-255) for the auto-torch feature.
   * When average frame brightness falls below this, the torch is activated.
   * Default: 60.
   */
  autoTorchThreshold?: number;

  /**
   * Run ZXing in parallel with OCR for damaged/ripped barcodes. Partial ZXing
   * digits + OCR text are merged to reconstruct the full barcode value.
   * Default: false.
   */
  enableDamagedBarcode?: boolean;

  /**
   * Enable NanoDet object-detection model for barcode region localization.
   * When false, ZXing runs directly on the full camera frame (faster, simpler).
   * Default: false.
   */
  enableNanoDet?: boolean;

  /**
   * Enable the consensus algorithm. Buffers the last N reads and only reports
   * a barcode when at least `consensusCount` identical values are seen.
   * Default: false.
   */
  enableConsensus?: boolean;

  /**
   * Number of identical reads required before the consensus algorithm reports
   * a barcode as confirmed. Default: 3.
   */
  consensusCount?: number;

  // ── Camera2 ISP settings (Android only) ─────────────────────────────────

  /**
   * Camera2 ISP configuration applied at the hardware level before frames
   * reach the barcode pipeline. Controls noise reduction, edge enhancement,
   * tonemapping, exposure compensation, and color correction.
   * Only effective on Android; ignored on iOS.
   *
   * Pass `'barcode'` for a barcode-optimized preset, `'high_quality'` for
   * max processing, or a custom ISPConfig object.
   */
  ispSettings?: 'barcode' | 'high_quality' | {
    noiseReduction?: 'off' | 'fast' | 'high_quality';
    edgeEnhancement?: 'off' | 'fast' | 'high_quality';
    tonemap?: 'fast' | 'high_quality' | 'gamma22' | 'srgb';
    exposureCompensation?: number;
    colorCorrection?: 'transform_matrix' | 'fast' | 'high_quality';
    shadingMode?: 'off' | 'fast' | 'high_quality';
    hotPixelMode?: 'off' | 'fast' | 'high_quality';
  };
}

export interface ZXingNanoDetFrameProcessorPlugin {
  detectBarcodes(
    frame: unknown,
    options?: DetectBarcodesOptions,
  ): BarcodeDetection[];
}
