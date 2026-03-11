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
  }, []);

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
