// SPDX-License-Identifier: Apache-2.0
// React hook that wires up the camera frame processor to collect barcode detections.

import { useCallback, useState } from 'react';
import { useFrameProcessor } from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { detectBarcodes } from './frameProcessor';
import type { BarcodeDetection, DetectBarcodesOptions } from './types';

export interface UseZXingNanoDetResult {
  /** Latest detections from the most-recently processed frame. */
  detections: BarcodeDetection[];
  /** Frame processor to pass to the VisionCamera <Camera> component. */
  frameProcessor: ReturnType<typeof useFrameProcessor>;
  /** Width of the camera sensor frame (landscape orientation). */
  frameWidth: number;
  /** Height of the camera sensor frame (landscape orientation). */
  frameHeight: number;
}

/**
 * Drop-in hook that returns both a VisionCamera `frameProcessor` and the
 * latest `detections` array derived from running NanoDet + ZXing on every frame.
 *
 * @example
 * ```tsx
 * const { frameProcessor, detections } = useZXingNanoDet({ confidence: 0.35 });
 * return <Camera device={device} isActive frameProcessor={frameProcessor} />;
 * ```
 */
export function useZXingNanoDet(
  options?: DetectBarcodesOptions,
): UseZXingNanoDetResult {
  const [detections, setDetections] = useState<BarcodeDetection[]>([]);
  const [frameWidth, setFrameWidth] = useState(0);
  const [frameHeight, setFrameHeight] = useState(0);

  const handleDetections = useCallback((results: BarcodeDetection[], fw: number, fh: number) => {
    setFrameWidth(fw);
    setFrameHeight(fh);
    setDetections(results);

    // Surface ALL native debug logs to the JS console so they are visible
    // without logcat. This includes C++ 1D decoder diagnostics, luma stats,
    // image quality metrics, ZXing decode attempts, and pipeline timing.
    if (options?.debug) {
      for (const det of results) {
        if (det.debugLogs && det.debugLogs.length > 0) {
          console.log(
            `[ZXingNanoDet] ${det.format} ${det.text ? `"${det.text}"` : '(no text)'} — ${det.debugLogs.length} log lines:`,
          );
          for (const line of det.debugLogs) {
            console.log(`  ${line}`);
          }
        }
        // Also log the detection itself with key data points
        if (det.format !== '__debug__') {
          console.log(
            `[ZXingNanoDet:DETECTION] format=${det.format} text="${det.text}" confidence=${det.confidence?.toFixed(3)} source=${det.source ?? 'unknown'} isOcr=${det.isOcrFallback ?? false} bbox=(${det.boundingBox?.x?.toFixed(0)},${det.boundingBox?.y?.toFixed(0)} ${det.boundingBox?.width?.toFixed(0)}x${det.boundingBox?.height?.toFixed(0)}) corners=${det.cornerPoints?.length ?? 0}`,
          );
        }
      }
      // Summary line even when no detections (for frame-level visibility)
      if (results.length === 0) {
        console.log(`[ZXingNanoDet] frame ${fw}x${fh}: no detections`);
      }
    }
  }, [options?.debug]);

  const runOnJsSetDetections = Worklets.createRunOnJS(handleDetections);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      // Capture frame dimensions synchronously BEFORE any async bridging
      const fw = frame.width;
      const fh = frame.height;
      const results = detectBarcodes(frame, options);
      runOnJsSetDetections(results, fw, fh);
    },
    [options, runOnJsSetDetections],
  );

  return { detections, frameProcessor, frameWidth, frameHeight };
}
