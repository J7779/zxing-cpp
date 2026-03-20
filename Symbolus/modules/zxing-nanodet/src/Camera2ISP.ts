// SPDX-License-Identifier: Apache-2.0
// TypeScript bridge for the Camera2ISP native module.
// Exposes Camera2 ISP settings (noise reduction, edge enhancement, tonemap,
// exposure compensation, color correction) to JS.

import { NativeModules, Platform } from 'react-native';

const { Camera2ISP } = NativeModules;

// ─── Types ──────────────────────────────────────────────────────────────────

export interface ISPConfig {
  /** Noise reduction: 'off' | 'fast' | 'high_quality' */
  noiseReduction?: 'off' | 'fast' | 'high_quality';
  /** Edge enhancement: 'off' | 'fast' | 'high_quality' */
  edgeEnhancement?: 'off' | 'fast' | 'high_quality';
  /** Tonemap mode: 'fast' | 'high_quality' | 'gamma22' | 'srgb' */
  tonemap?: 'fast' | 'high_quality' | 'gamma22' | 'srgb';
  /** Exposure compensation in EV stops, e.g. -2 to +2. Clamped to device range. */
  exposureCompensation?: number;
  /** Color correction: 'transform_matrix' | 'fast' | 'high_quality' */
  colorCorrection?: 'transform_matrix' | 'fast' | 'high_quality';
  /** Lens shading: 'off' | 'fast' | 'high_quality' */
  shadingMode?: 'off' | 'fast' | 'high_quality';
  /** Hot pixel correction: 'off' | 'fast' | 'high_quality' */
  hotPixelMode?: 'off' | 'fast' | 'high_quality';
}

export interface ISPCapabilities {
  noiseReductionModes: string[];
  edgeModes: string[];
  tonemapModes: string[];
  exposureCompensationRange: [number, number];
  exposureCompensationStep: number;
  colorCorrectionModes: string[];
  shadingModes: string[];
  hotPixelModes: string[];
}

// ─── API ────────────────────────────────────────────────────────────────────

/**
 * Apply Camera2 ISP settings to the active CameraX camera session.
 * Settings take effect on the next capture request (essentially immediately).
 * Only works on Android; no-ops gracefully on other platforms.
 */
export async function applyISPSettings(config: ISPConfig): Promise<void> {
  if (Platform.OS !== 'android' || !Camera2ISP) return;
  return Camera2ISP.applyISPSettings(config);
}

/**
 * Query which ISP modes the current camera hardware supports.
 * Returns null on non-Android platforms.
 */
export async function getISPCapabilities(): Promise<ISPCapabilities | null> {
  if (Platform.OS !== 'android' || !Camera2ISP) return null;
  return Camera2ISP.getISPCapabilities();
}

/**
 * Reset all ISP settings to camera defaults (auto everything).
 */
export async function resetISPSettings(): Promise<void> {
  if (Platform.OS !== 'android' || !Camera2ISP) return;
  return Camera2ISP.resetISPSettings();
}

/** Barcode-optimized ISP preset: fast noise reduction, fast edge, gamma 2.2, +0.5 EV. */
export const ISP_BARCODE_PRESET: ISPConfig = {
  noiseReduction: 'fast',
  edgeEnhancement: 'fast',
  tonemap: 'gamma22',
  exposureCompensation: 0.5,
  colorCorrection: 'fast',
};

/** High-quality ISP preset: maximum processing for difficult conditions. */
export const ISP_HIGH_QUALITY_PRESET: ISPConfig = {
  noiseReduction: 'high_quality',
  edgeEnhancement: 'high_quality',
  tonemap: 'high_quality',
  exposureCompensation: 0,
  colorCorrection: 'high_quality',
};
