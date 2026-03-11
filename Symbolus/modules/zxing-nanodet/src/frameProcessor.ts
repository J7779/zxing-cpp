// SPDX-License-Identifier: Apache-2.0
// Registers the native frame processor plugin with VisionCamera.
// The native side (iOS .mm / Android .kt) must register a plugin
// named "detectBarcodes" via their respective registries.

import { VisionCameraProxy } from 'react-native-vision-camera';
import type { BarcodeDetection, DetectBarcodesOptions } from './types';

// VisionCamera v4 uses VisionCameraProxy.initFrameProcessorPlugin.
const plugin = VisionCameraProxy.initFrameProcessorPlugin('detectBarcodes', {});

if (!plugin) {
  console.warn(
    '[zxing-nanodet] Native "detectBarcodes" frame processor plugin not found. ' +
    'Make sure react-native-worklets-core is installed, babel.config.js includes ' +
    'the worklets plugin, and you ran `npx expo prebuild`.',
  );
}

/**
 * Worklet-safe frame processor function.
 * Call this inside a `useFrameProcessor` worklet on each camera frame.
 *
 * @example
 * ```ts
 * const frameProcessor = useFrameProcessor((frame) => {
 *   'worklet';
 *   const results = detectBarcodes(frame, { confidence: 0.35 });
 *   runOnJS(setDetections)(results);
 * }, []);
 * ```
 */
export function detectBarcodes(
  frame: unknown,
  options?: DetectBarcodesOptions,
): BarcodeDetection[] {
  'worklet';
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (plugin as any)?.call(frame, options) ?? [];
}
