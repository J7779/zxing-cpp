// SPDX-License-Identifier: Apache-2.0
// ScannerSettingsContext.tsx
//
// Global scanner settings: engine toggles, per-format enables, lighting,
// parallel scan, damaged-barcode merge, consensus algorithm.
// Defaults to safe values. Wrap the app root with <ScannerSettingsProvider>.

import React, { createContext, useCallback, useContext, useState } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// Master format list — exact strings returned by ZXing-C++ ToString()
// ─────────────────────────────────────────────────────────────────────────────

export type FormatGroup = {
  label: string;
  formats: readonly FormatEntry[];
};

export type FormatEntry = {
  /** Exact ZXing-C++ format string (used for filtering) */
  id: string;
  /** Human-readable display name */
  label: string;
};

export const FORMAT_GROUPS: FormatGroup[] = [
  {
    label: '1D Linear',
    formats: [
      { id: 'Code128',          label: 'Code 128' },
      { id: 'Code39',           label: 'Code 39' },
      { id: 'Code93',           label: 'Code 93' },
      { id: 'Codabar',          label: 'Codabar' },
      { id: 'ITF',              label: 'ITF (Interleaved 2-of-5)' },
      { id: 'DataBar',          label: 'DataBar (GS1)' },
      { id: 'DataBarExpanded',  label: 'DataBar Expanded' },
      { id: 'DataBarLimited',   label: 'DataBar Limited' },
    ],
  },
  {
    label: 'EAN / UPC',
    formats: [
      { id: 'EAN-13', label: 'EAN-13' },
      { id: 'EAN-8',  label: 'EAN-8' },
      { id: 'UPC-A',  label: 'UPC-A' },
      { id: 'UPC-E',  label: 'UPC-E' },
    ],
  },
  {
    label: '2D Matrix',
    formats: [
      { id: 'QRCode',       label: 'QR Code' },
      { id: 'DataMatrix',   label: 'Data Matrix' },
      { id: 'PDF417',       label: 'PDF 417' },
      { id: 'Aztec',        label: 'Aztec' },
      { id: 'MicroQRCode',  label: 'Micro QR Code' },
      { id: 'rMQRCode',     label: 'rMQR Code' },
      { id: 'MaxiCode',     label: 'MaxiCode' },
    ],
  },
];

export const ALL_FORMAT_IDS: string[] = FORMAT_GROUPS.flatMap((g) =>
  g.formats.map((f) => f.id),
);

// ─────────────────────────────────────────────────────────────────────────────
// Torch / lighting types
// ─────────────────────────────────────────────────────────────────────────────

export type TorchMode = 'off' | 'on' | 'auto';

// ─────────────────────────────────────────────────────────────────────────────
// Context types
// ─────────────────────────────────────────────────────────────────────────────

export interface ScannerSettings {
  // Engines
  enableZxing: boolean;
  enableOcr: boolean;
  /** Set of active format ids — filter applied in Kotlin after ZXing decodes */
  enabledFormats: ReadonlySet<string>;

  // Lighting
  torchMode: TorchMode;
  autoTorchThreshold: number;

  // Parallel direct-ZXing on scan-box region (bypasses NanoDet)
  enableDirectZxing: boolean;

  // ZXing resolution multiplier (>1 = upscale crop for better decode)
  zxingResolutionScale: number;

  // Damaged/ripped barcode: merge partial ZXing + OCR reads
  enableDamagedBarcode: boolean;

  // Consensus algorithm
  enableConsensus: boolean;
  consensusCount: number;
}

interface ScannerSettingsContextValue {
  settings: ScannerSettings;
  setEnableZxing: (v: boolean) => void;
  setEnableOcr: (v: boolean) => void;
  toggleFormat: (id: string) => void;
  setAllFormats: (enabled: boolean) => void;
  setTorchMode: (v: TorchMode) => void;
  setAutoTorchThreshold: (v: number) => void;
  setEnableDirectZxing: (v: boolean) => void;
  setZxingResolutionScale: (v: number) => void;
  setEnableDamagedBarcode: (v: boolean) => void;
  setEnableConsensus: (v: boolean) => void;
  setConsensusCount: (v: number) => void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Defaults
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_SETTINGS: ScannerSettings = {
  enableZxing: true,
  enableOcr: true,
  enabledFormats: new Set(ALL_FORMAT_IDS),
  torchMode: 'off',
  autoTorchThreshold: 60,
  enableDirectZxing: false,
  zxingResolutionScale: 1.0,
  enableDamagedBarcode: false,
  enableConsensus: false,
  consensusCount: 3,
};

// ─────────────────────────────────────────────────────────────────────────────
// Context
// ─────────────────────────────────────────────────────────────────────────────

const ScannerSettingsContext = createContext<ScannerSettingsContextValue>({
  settings: DEFAULT_SETTINGS,
  setEnableZxing: () => {},
  setEnableOcr: () => {},
  toggleFormat: () => {},
  setAllFormats: () => {},
  setTorchMode: () => {},
  setAutoTorchThreshold: () => {},
  setEnableDirectZxing: () => {},
  setZxingResolutionScale: () => {},
  setEnableDamagedBarcode: () => {},
  setEnableConsensus: () => {},
  setConsensusCount: () => {},
});

export function ScannerSettingsProvider({ children }: { children: React.ReactNode }) {
  const [enableZxing, setEnableZxing] = useState(DEFAULT_SETTINGS.enableZxing);
  const [enableOcr, setEnableOcr] = useState(DEFAULT_SETTINGS.enableOcr);
  const [enabledFormats, setEnabledFormats] = useState<Set<string>>(
    () => new Set(ALL_FORMAT_IDS),
  );
  const [torchMode, setTorchMode] = useState<TorchMode>(DEFAULT_SETTINGS.torchMode);
  const [autoTorchThreshold, setAutoTorchThreshold] = useState(DEFAULT_SETTINGS.autoTorchThreshold);
  const [enableDirectZxing, setEnableDirectZxing] = useState(DEFAULT_SETTINGS.enableDirectZxing);
  const [zxingResolutionScale, setZxingResolutionScale] = useState(DEFAULT_SETTINGS.zxingResolutionScale);
  const [enableDamagedBarcode, setEnableDamagedBarcode] = useState(DEFAULT_SETTINGS.enableDamagedBarcode);
  const [enableConsensus, setEnableConsensus] = useState(DEFAULT_SETTINGS.enableConsensus);
  const [consensusCount, setConsensusCount] = useState(DEFAULT_SETTINGS.consensusCount);

  const toggleFormat = useCallback((id: string) => {
    setEnabledFormats((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const setAllFormats = useCallback((enabled: boolean) => {
    setEnabledFormats(enabled ? new Set(ALL_FORMAT_IDS) : new Set<string>());
  }, []);

  return (
    <ScannerSettingsContext.Provider
      value={{
        settings: {
          enableZxing, enableOcr, enabledFormats,
          torchMode, autoTorchThreshold,
          enableDirectZxing, zxingResolutionScale,
          enableDamagedBarcode, enableConsensus, consensusCount,
        },
        setEnableZxing, setEnableOcr,
        toggleFormat, setAllFormats,
        setTorchMode, setAutoTorchThreshold,
        setEnableDirectZxing, setZxingResolutionScale,
        setEnableDamagedBarcode, setEnableConsensus, setConsensusCount,
      }}
    >
      {children}
    </ScannerSettingsContext.Provider>
  );
}

export function useScannerSettings(): ScannerSettingsContextValue {
  return useContext(ScannerSettingsContext);
}
