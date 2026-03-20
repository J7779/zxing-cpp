// SPDX-License-Identifier: Apache-2.0
// JS entry-point — exports the frame processor plugin and helpers

export type {
  BarcodeDetection,
  DetectBarcodesOptions,
} from './types';

export { useZXingNanoDet } from './useZXingNanoDet';
export { detectBarcodes } from './frameProcessor';

// Camera2 ISP control (Android only)
export {
  applyISPSettings,
  getISPCapabilities,
  resetISPSettings,
  ISP_BARCODE_PRESET,
  ISP_HIGH_QUALITY_PRESET,
} from './Camera2ISP';
export type { ISPConfig, ISPCapabilities } from './Camera2ISP';

// Native barcode overlay (Android only)
export { BarcodeOverlay } from './BarcodeOverlay';
