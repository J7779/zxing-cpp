// SPDX-License-Identifier: Apache-2.0
// JS entry-point — exports the frame processor plugin and helpers

export type {
  BarcodeDetection,
  DetectBarcodesOptions,
} from './types';

export { useZXingNanoDet } from './useZXingNanoDet';
export { detectBarcodes } from './frameProcessor';
