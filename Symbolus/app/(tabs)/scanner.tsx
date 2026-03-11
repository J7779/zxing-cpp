// SPDX-License-Identifier: Apache-2.0
// app/(tabs)/scanner.tsx — Barcode scanner tab screen

import React, { useCallback, useMemo, useState } from 'react';
import {
  Animated,
  FlatList,
  Platform,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import BarcodeScanner from '@/components/BarcodeScanner';
import { ThemedText } from '@/components/themed-text';
import { useScannerSettings, ALL_FORMAT_IDS } from '@/contexts/ScannerSettingsContext';
import type { BarcodeDetection } from '../../modules/zxing-nanodet/src/types';

// ─────────────────────────────────────────────────────────────────────────────
// History entry
// ─────────────────────────────────────────────────────────────────────────────

interface HistoryItem {
  id: string;
  timestamp: number;
  format: string;
  text: string;
  confidence: number;
  boundingBox: BarcodeDetection['boundingBox'];
  cornerPoints?: BarcodeDetection['cornerPoints'];
}

function HistoryRow({ item }: { item: HistoryItem }) {
  const date = new Date(item.timestamp);
  const timeStr = date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });

  return (
    <View style={styles.historyRow}>
      <View style={styles.historyLeft}>
        <Text style={styles.historyFormat}>{item.format}</Text>
        <Text style={styles.historyText} numberOfLines={2}>{item.text}</Text>
      </View>
      <Text style={styles.historyTime}>{timeStr}</Text>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Screen
// ─────────────────────────────────────────────────────────────────────────────

export default function ScannerScreen() {
  const insets = useSafeAreaInsets();
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [panelOpen, setPanelOpen] = useState(false);
  const { settings } = useScannerSettings();

  // Build detection options from settings; recompute only when settings change.
  const detectionOptions = useMemo(() => ({
    confidence: 0.35,
    modelInputSize: 640,
    maxDetections: 10,
    enableZxing: settings.enableZxing,
    enableOcr: settings.enableOcr,
    // When all formats enabled pass undefined (no filter) — avoids passing a huge array every frame.
    enabledFormats:
      settings.enabledFormats.size === ALL_FORMAT_IDS.length
        ? undefined
        : Array.from(settings.enabledFormats),
  }), [settings]);

  const handleDetected = useCallback((det: BarcodeDetection) => {
    setHistory((prev) => {
      // Deduplicate: skip if the same text was already added within 2 s
      const last = prev[0];
      if (last && last.text === det.text && Date.now() - last.timestamp < 2000) {
        return prev;
      }
      const item: HistoryItem = {
        id: `${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        format: det.format,
        text: det.text,
        confidence: det.confidence,
        boundingBox: det.boundingBox,
        cornerPoints: det.cornerPoints,
      };
      return [item, ...prev].slice(0, 50); // keep last 50
    });
    setPanelOpen(true);
  }, []);

  return (
    <View style={styles.container}>
      {/* ── Camera ─────────────────────────────────────────────────────── */}
      <BarcodeScanner
        style={panelOpen ? styles.cameraSmall : styles.cameraFull}
        onBarcodeDetected={handleDetected}
        detectionOptions={detectionOptions}
        cooldownMs={1500}
      >
        {/* Top bar */}
        <View style={[styles.topBar, { paddingTop: insets.top + 8 }]}>
          <ThemedText type="title" style={styles.topBarTitle}>
            Symbolus Scanner
          </ThemedText>
          {history.length > 0 && (
            <TouchableOpacity
              style={styles.historyToggle}
              onPress={() => setPanelOpen((v) => !v)}
            >
              <Text style={styles.historyToggleText}>
                {panelOpen ? 'Hide' : `History (${history.length})`}
              </Text>
            </TouchableOpacity>
          )}
        </View>
      </BarcodeScanner>

      {/* ── History panel ──────────────────────────────────────────────── */}
      {panelOpen && history.length > 0 && (
        <View style={[styles.historyPanel, { paddingBottom: insets.bottom }]}>
          <View style={styles.historyHeader}>
            <Text style={styles.historyHeaderTitle}>Scan History</Text>
            <TouchableOpacity onPress={() => setHistory([])}>
              <Text style={styles.clearBtn}>Clear</Text>
            </TouchableOpacity>
          </View>
          <FlatList
            data={history}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => <HistoryRow item={item} />}
            ItemSeparatorComponent={() => <View style={styles.separator} />}
          />
        </View>
      )}

      {/* Empty state hint when panel closed and no history */}
      {!panelOpen && history.length === 0 && (
        <View style={[styles.hint, { paddingBottom: insets.bottom + 12 }]}>
          <Text style={styles.hintText}>
            Detected barcodes will appear here
          </Text>
        </View>
      )}
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────────────────────────────────────

const PANEL_HEIGHT = 280;

const styles = StyleSheet.create({
  container: {
    flex:            1,
    backgroundColor: '#000',
  },

  // ── Camera ────────────────────────────────────────────────────────────────
  cameraFull: {
    flex: 1,
  },
  cameraSmall: {
    flex:   0,
    height: '55%' as unknown as number,
  },

  // ── Top bar ──────────────────────────────────────────────────────────────
  topBar: {
    position:        'absolute',
    top:             0,
    left:            0,
    right:           0,
    flexDirection:   'row',
    alignItems:      'center',
    justifyContent:  'space-between',
    paddingHorizontal: 16,
    paddingBottom:   10,
    backgroundColor: 'rgba(0,0,0,0.45)',
  },
  topBarTitle: {
    color:      '#FFFFFF',
    fontSize:   18,
    fontWeight: '700',
  },
  historyToggle: {
    backgroundColor: 'rgba(255,255,255,0.15)',
    paddingHorizontal: 12,
    paddingVertical:    6,
    borderRadius: 14,
  },
  historyToggleText: {
    color:      '#00E5FF',
    fontSize:   13,
    fontWeight: '600',
  },

  // ── History panel ─────────────────────────────────────────────────────────
  historyPanel: {
    backgroundColor: '#111',
    maxHeight:       PANEL_HEIGHT,
    borderTopWidth:  1,
    borderTopColor:  '#333',
  },
  historyHeader: {
    flexDirection:     'row',
    alignItems:        'center',
    justifyContent:    'space-between',
    paddingHorizontal: 16,
    paddingVertical:   10,
    borderBottomWidth: 1,
    borderBottomColor: '#222',
  },
  historyHeaderTitle: {
    color:      '#FFF',
    fontSize:   15,
    fontWeight: '600',
  },
  clearBtn: {
    color:    '#FF6B6B',
    fontSize: 14,
  },
  historyRow: {
    flexDirection:     'row',
    alignItems:        'center',
    paddingHorizontal: 16,
    paddingVertical:   10,
  },
  historyLeft: {
    flex: 1,
  },
  historyFormat: {
    color:      '#00E5FF',
    fontSize:   11,
    fontWeight: '600',
    letterSpacing: 0.5,
    marginBottom: 2,
  },
  historyText: {
    color:    '#EEE',
    fontSize: 14,
  },
  historyTime: {
    color:    '#888',
    fontSize: 11,
    marginLeft: 10,
  },
  separator: {
    height:          1,
    backgroundColor: '#1E1E1E',
    marginHorizontal: 16,
  },

  // ── Hint ──────────────────────────────────────────────────────────────────
  hint: {
    alignItems:      'center',
    padding:         12,
    backgroundColor: 'rgba(0,0,0,0.6)',
  },
  hintText: {
    color:    'rgba(255,255,255,0.45)',
    fontSize: 13,
  },
});
