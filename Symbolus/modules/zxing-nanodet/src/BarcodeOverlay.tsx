// SPDX-License-Identifier: Apache-2.0
// React Native bridge for the native BarcodeOverlayView (Android).
// Renders ZXing bounding boxes and corner points natively via Canvas.
// The overlay updates automatically from the frame processor plugin.

import { Platform, StyleSheet, requireNativeComponent, type ViewProps } from 'react-native';
import React from 'react';

interface NativeOverlayProps extends ViewProps {
  visible?: boolean;
  mirrorX?: boolean;
}

const NativeBarcodeOverlay =
  Platform.OS === 'android'
    ? requireNativeComponent<NativeOverlayProps>('BarcodeOverlayView')
    : null;

/**
 * Native overlay that draws ZXing barcode bounding boxes and corner points.
 * Place this as a sibling of the Camera view, absolutely positioned on top.
 * The overlay receives detection data internally from the frame processor
 * plugin — no JS props needed for detection data.
 *
 * @example
 * ```tsx
 * <Camera ... />
 * <BarcodeOverlay style={StyleSheet.absoluteFill} />
 * ```
 */
export function BarcodeOverlay({
  visible = true,
  mirrorX = false,
  style,
  ...rest
}: NativeOverlayProps) {
  if (!NativeBarcodeOverlay) return null;
  return (
    <NativeBarcodeOverlay
      visible={visible}
      mirrorX={mirrorX}
      style={[StyleSheet.absoluteFill, style]}
      {...rest}
    />
  );
}
