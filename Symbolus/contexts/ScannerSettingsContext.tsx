// SPDX-License-Identifier: Apache-2.0
// ScannerSettingsContext.tsx
//
// Global scanner settings: engine toggles (ZXing / OCR) and per-format enables.
// Defaults to everything on. Wrap the app root with <ScannerSettingsProvider>.

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
// Context types
// ─────────────────────────────────────────────────────────────────────────────

export interface ScannerSettings {
  enableZxing: boolean;
  enableOcr: boolean;
  /** Set of active format ids — filter applied in Kotlin after ZXing decodes */
  enabledFormats: ReadonlySet<string>;
}

interface ScannerSettingsContextValue {
  settings: ScannerSettings;
  setEnableZxing: (v: boolean) => void;
  setEnableOcr: (v: boolean) => void;
  toggleFormat: (id: string) => void;
  setAllFormats: (enabled: boolean) => void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Context
// ─────────────────────────────────────────────────────────────────────────────

const ScannerSettingsContext = createContext<ScannerSettingsContextValue>({
  settings: {
    enableZxing: true,
    enableOcr: true,
    enabledFormats: new Set(ALL_FORMAT_IDS),
  },
  setEnableZxing: () => {},
  setEnableOcr: () => {},
  toggleFormat: () => {},
  setAllFormats: () => {},
});

export function ScannerSettingsProvider({ children }: { children: React.ReactNode }) {
  const [enableZxing, setEnableZxing] = useState(true);
  const [enableOcr, setEnableOcr] = useState(true);
  const [enabledFormats, setEnabledFormats] = useState<Set<string>>(
    () => new Set(ALL_FORMAT_IDS),
  );

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
        settings: { enableZxing, enableOcr, enabledFormats },
        setEnableZxing,
        setEnableOcr,
        toggleFormat,
        setAllFormats,
      }}
    >
      {children}
    </ScannerSettingsContext.Provider>
  );
}

export function useScannerSettings(): ScannerSettingsContextValue {
  return useContext(ScannerSettingsContext);
}
