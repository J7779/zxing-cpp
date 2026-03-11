// SPDX-License-Identifier: Apache-2.0
// app/(tabs)/settings.tsx — Scanner settings screen

import React from 'react';
import {
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import {
  FORMAT_GROUPS,
  ALL_FORMAT_IDS,
  useScannerSettings,
  type TorchMode,
} from '@/contexts/ScannerSettingsContext';

// ─────────────────────────────────────────────────────────────────────────────
// Reusable rows
// ─────────────────────────────────────────────────────────────────────────────

function SettingsRow({
  label,
  sublabel,
  value,
  onValueChange,
  trackColor,
}: {
  label: string;
  sublabel?: string;
  value: boolean;
  onValueChange: (v: boolean) => void;
  trackColor?: { false: string; true: string };
}) {
  return (
    <View style={styles.row}>
      <View style={styles.rowText}>
        <Text style={styles.rowLabel}>{label}</Text>
        {sublabel ? <Text style={styles.rowSublabel}>{sublabel}</Text> : null}
      </View>
      <Switch
        value={value}
        onValueChange={onValueChange}
        trackColor={trackColor ?? { false: '#3A3A3A', true: '#00B4CC' }}
        thumbColor={value ? '#FFFFFF' : '#888'}
        ios_backgroundColor="#3A3A3A"
      />
    </View>
  );
}

function SegmentRow<T extends string>({
  label,
  sublabel,
  options,
  value,
  onValueChange,
}: {
  label: string;
  sublabel?: string;
  options: { label: string; value: T }[];
  value: T;
  onValueChange: (v: T) => void;
}) {
  return (
    <View style={styles.row}>
      <View style={[styles.rowText, { flex: 1, marginRight: 8 }]}>
        <Text style={styles.rowLabel}>{label}</Text>
        {sublabel ? <Text style={styles.rowSublabel}>{sublabel}</Text> : null}
      </View>
      <View style={styles.segmentGroup}>
        {options.map((o) => (
          <TouchableOpacity
            key={o.value}
            style={[styles.segmentBtn, value === o.value && styles.segmentBtnActive]}
            onPress={() => onValueChange(o.value)}
          >
            <Text style={[styles.segmentBtnText, value === o.value && styles.segmentBtnTextActive]}>
              {o.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

function SectionHeader({ title }: { title: string }) {
  return <Text style={styles.sectionHeader}>{title}</Text>;
}

function Card({ children }: { children: React.ReactNode }) {
  return <View style={styles.card}>{children}</View>;
}

function Divider() {
  return <View style={styles.divider} />;
}

// ─────────────────────────────────────────────────────────────────────────────
// Screen
// ─────────────────────────────────────────────────────────────────────────────

export default function SettingsScreen() {
  const insets = useSafeAreaInsets();
  const {
    settings,
    setEnableZxing, setEnableOcr,
    toggleFormat, setAllFormats,
    setTorchMode, setAutoTorchThreshold,
    setEnableDirectZxing, setZxingResolutionScale,
    setEnableDamagedBarcode,
    setEnableConsensus, setConsensusCount,
  } = useScannerSettings();

  const allOn = ALL_FORMAT_IDS.every((id) => settings.enabledFormats.has(id));
  const allOff = ALL_FORMAT_IDS.every((id) => !settings.enabledFormats.has(id));

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={[
        styles.content,
        { paddingTop: insets.top + 16, paddingBottom: insets.bottom + 24 },
      ]}
      showsVerticalScrollIndicator={false}
    >
      <Text style={styles.title}>Scanner Settings</Text>

      {/* ── Detection engines ──────────────────────────────────────────── */}
      <SectionHeader title="DETECTION ENGINES" />
      <Card>
        <SettingsRow
          label="ZXing Decode"
          sublabel="Structured barcode decode (fast, precise)"
          value={settings.enableZxing}
          onValueChange={setEnableZxing}
          trackColor={{ false: '#3A3A3A', true: '#00B4CC' }}
        />
        <Divider />
        <SettingsRow
          label="OCR Fallback (PP-OCRv5)"
          sublabel="Text recognition when ZXing fails"
          value={settings.enableOcr}
          onValueChange={setEnableOcr}
          trackColor={{ false: '#3A3A3A', true: '#FFB300' }}
        />
      </Card>

      {/* ── Lighting ───────────────────────────────────────────────────── */}
      <SectionHeader title="LIGHTING" />
      <Card>
        <SegmentRow<TorchMode>
          label="Flash / Torch"
          sublabel="Helps in dark environments"
          options={[
            { label: 'Off', value: 'off' },
            { label: 'On', value: 'on' },
            { label: 'Auto', value: 'auto' },
          ]}
          value={settings.torchMode}
          onValueChange={setTorchMode}
        />
        {settings.torchMode === 'auto' && (
          <>
            <Divider />
            <SegmentRow<string>
              label="Auto Threshold"
              sublabel="Brightness level to trigger torch"
              options={[
                { label: 'Low', value: '40' },
                { label: 'Med', value: '60' },
                { label: 'High', value: '80' },
              ]}
              value={String(settings.autoTorchThreshold)}
              onValueChange={(v) => setAutoTorchThreshold(Number(v))}
            />
          </>
        )}
      </Card>

      {/* ── Advanced scanning ──────────────────────────────────────────── */}
      <SectionHeader title="ADVANCED SCANNING" />
      <Card>
        <SettingsRow
          label="Direct ZXing Scan"
          sublabel="Run ZXing on full scan area in parallel with NanoDet"
          value={settings.enableDirectZxing}
          onValueChange={setEnableDirectZxing}
          trackColor={{ false: '#3A3A3A', true: '#4CAF50' }}
        />
        <Divider />
        <SegmentRow<string>
          label="ZXing Resolution"
          sublabel="Higher = better decode but slower"
          options={[
            { label: '1x', value: '1' },
            { label: '1.5x', value: '1.5' },
            { label: '2x', value: '2' },
          ]}
          value={String(settings.zxingResolutionScale)}
          onValueChange={(v) => setZxingResolutionScale(Number(v))}
        />
        <Divider />
        <SettingsRow
          label="Damaged Barcode Merge"
          sublabel="Combine partial ZXing + OCR for ripped barcodes"
          value={settings.enableDamagedBarcode}
          onValueChange={setEnableDamagedBarcode}
          trackColor={{ false: '#3A3A3A', true: '#FF7043' }}
        />
      </Card>

      {/* ── Consensus ──────────────────────────────────────────────────── */}
      <SectionHeader title="ACCURACY" />
      <Card>
        <SettingsRow
          label="Consensus Algorithm"
          sublabel="Require multiple consistent reads before reporting"
          value={settings.enableConsensus}
          onValueChange={setEnableConsensus}
          trackColor={{ false: '#3A3A3A', true: '#AB47BC' }}
        />
        {settings.enableConsensus && (
          <>
            <Divider />
            <SegmentRow<string>
              label="Required Reads"
              sublabel="Number of identical reads to confirm"
              options={[
                { label: '2', value: '2' },
                { label: '3', value: '3' },
                { label: '5', value: '5' },
              ]}
              value={String(settings.consensusCount)}
              onValueChange={(v) => setConsensusCount(Number(v))}
            />
          </>
        )}
      </Card>

      {/* ── Barcode formats ────────────────────────────────────────────── */}
      <SectionHeader title="BARCODE FORMATS" />

      {/* Select all / none */}
      <View style={styles.bulkRow}>
        <TouchableOpacity
          style={[styles.bulkBtn, allOn && styles.bulkBtnActive]}
          onPress={() => setAllFormats(true)}
        >
          <Text style={[styles.bulkBtnText, allOn && styles.bulkBtnTextActive]}>
            All On
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.bulkBtn, allOff && styles.bulkBtnActive]}
          onPress={() => setAllFormats(false)}
        >
          <Text style={[styles.bulkBtnText, allOff && styles.bulkBtnTextActive]}>
            All Off
          </Text>
        </TouchableOpacity>
      </View>

      {FORMAT_GROUPS.map((group) => (
        <React.Fragment key={group.label}>
          <Text style={styles.groupLabel}>{group.label}</Text>
          <Card>
            {group.formats.map((fmt, idx) => (
              <React.Fragment key={fmt.id}>
                {idx > 0 && <Divider />}
                <SettingsRow
                  label={fmt.label}
                  value={settings.enabledFormats.has(fmt.id)}
                  onValueChange={() => toggleFormat(fmt.id)}
                />
              </React.Fragment>
            ))}
          </Card>
        </React.Fragment>
      ))}
    </ScrollView>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0A0A',
  },
  content: {
    paddingHorizontal: 16,
  },
  title: {
    color: '#FFFFFF',
    fontSize: 26,
    fontWeight: '700',
    marginBottom: 20,
  },

  sectionHeader: {
    color: '#666',
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: 1.2,
    marginTop: 20,
    marginBottom: 8,
    marginLeft: 4,
  },

  groupLabel: {
    color: '#888',
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 0.8,
    marginTop: 14,
    marginBottom: 6,
    marginLeft: 4,
  },

  card: {
    backgroundColor: '#1A1A1A',
    borderRadius: 12,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#2A2A2A',
  },

  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    minHeight: 52,
  },
  rowText: {
    flex: 1,
    marginRight: 12,
  },
  rowLabel: {
    color: '#EEEEEE',
    fontSize: 15,
    fontWeight: '500',
  },
  rowSublabel: {
    color: '#888',
    fontSize: 12,
    marginTop: 2,
  },

  divider: {
    height: 1,
    backgroundColor: '#2A2A2A',
    marginLeft: 16,
  },

  bulkRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 4,
  },
  bulkBtn: {
    flex: 1,
    paddingVertical: 9,
    borderRadius: 8,
    alignItems: 'center',
    backgroundColor: '#1A1A1A',
    borderWidth: 1,
    borderColor: '#2A2A2A',
  },
  bulkBtnActive: {
    backgroundColor: '#00B4CC22',
    borderColor: '#00B4CC',
  },
  bulkBtnText: {
    color: '#888',
    fontSize: 14,
    fontWeight: '600',
  },
  bulkBtnTextActive: {
    color: '#00B4CC',
  },

  // Segment control (for torch mode, resolution picker, etc.)
  segmentGroup: {
    flexDirection: 'row',
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#2A2A2A',
  },
  segmentBtn: {
    paddingHorizontal: 12,
    paddingVertical: 7,
    backgroundColor: '#1A1A1A',
  },
  segmentBtnActive: {
    backgroundColor: '#00B4CC',
  },
  segmentBtnText: {
    color: '#888',
    fontSize: 13,
    fontWeight: '600',
  },
  segmentBtnTextActive: {
    color: '#FFFFFF',
  },
});
