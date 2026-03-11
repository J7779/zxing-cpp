// SPDX-License-Identifier: Apache-2.0
// BarcodeScanner.tsx
//
// Full-screen camera view with:
//  • Scan-region box (Scandit-style) — only detections inside the box are processed
//  • Live NanoDet bounding-box overlays
//  • ZXing-decoded text label beneath each box
//  • Pinch-to-zoom, tap-to-focus
//  • Haptic + callback when a new barcode is confirmed

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactElement,
} from 'react';
import {
  Linking,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Image,
  type ViewStyle,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraFormat,
  useCameraPermission,
  type CameraDevice,
} from 'react-native-vision-camera';
import { GestureHandlerRootView, Gesture, GestureDetector } from 'react-native-gesture-handler';
import { runOnJS, useSharedValue } from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';

import {
  useZXingNanoDet,
  type BarcodeDetection,
  type DetectBarcodesOptions,
} from '../modules/zxing-nanodet/src';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// Scan region: square box (works for both barcodes and QR codes).
// Side length = 82% of whichever container axis is smaller → always square.
const SCAN_REGION_FRAC = 0.82;

// ─────────────────────────────────────────────────────────────────────────────
// Props
// ─────────────────────────────────────────────────────────────────────────────

export interface BarcodeScannerProps {
  /** Called once per unique barcode text detected (debounced, deduped). */
  onBarcodeDetected?: (detection: BarcodeDetection) => void;
  /** Override NanoDet + ZXing detection options. */
  detectionOptions?: DetectBarcodesOptions;
  /** How long (ms) to wait before reporting the same barcode again. Default 2000. */
  cooldownMs?: number;
  /** Render extra UI on top of the camera (e.g. a close button). */
  children?: ReactElement | ReactElement[];
  style?: ViewStyle;
  /** Which camera to use — defaults to 'back'. */
  facing?: 'front' | 'back';
}

// ─────────────────────────────────────────────────────────────────────────────
// Detection overlay
// ─────────────────────────────────────────────────────────────────────────────

interface OverlayBoxProps {
  detection: BarcodeDetection;
  frameWidth: number;
  frameHeight: number;
  containerWidth: number;
  containerHeight: number;
}

function OverlayBox({
  detection,
  frameWidth,
  frameHeight,
  containerWidth,
  containerHeight,
}: OverlayBoxProps) {
  const { boundingBox, text, format, confidence, isOcrFallback } = detection;
  const borderColor = isOcrFallback ? OCR_BORDER_COLOR : BORDER_COLOR;

  // Detect rotation: sensor frame is landscape (w>h) but container is portrait (h>w)
  const needsRotation = frameWidth > frameHeight && containerHeight > containerWidth;

  let displayX: number, displayY: number, displayW: number, displayH: number;
  let displayFrameW: number, displayFrameH: number;

  if (needsRotation) {
    // 90° CW rotation: sensor landscape → display portrait
    displayX = frameHeight - boundingBox.y - boundingBox.height;
    displayY = boundingBox.x;
    displayW = boundingBox.height;
    displayH = boundingBox.width;
    displayFrameW = frameHeight;
    displayFrameH = frameWidth;
  } else {
    displayX = boundingBox.x;
    displayY = boundingBox.y;
    displayW = boundingBox.width;
    displayH = boundingBox.height;
    displayFrameW = frameWidth;
    displayFrameH = frameHeight;
  }

  const scaleX = containerWidth  / displayFrameW;
  const scaleY = containerHeight / displayFrameH;

  const left   = displayX * scaleX;
  const top    = displayY * scaleY;
  const width  = displayW * scaleX;
  const height = displayH * scaleY;

  console.log(
    `[OverlayBox] format=${format} text="${text?.substring(0, 30)}" conf=${confidence}\n` +
    `  boundingBox: x=${boundingBox.x} y=${boundingBox.y} w=${boundingBox.width} h=${boundingBox.height}\n` +
    `  frame: ${frameWidth}x${frameHeight} container: ${containerWidth}x${containerHeight}\n` +
    `  scale: scaleX=${scaleX.toFixed(4)} scaleY=${scaleY.toFixed(4)}\n` +
    `  rendered: left=${left.toFixed(1)} top=${top.toFixed(1)} width=${width.toFixed(1)} height=${height.toFixed(1)}`,
  );

  return (
    <View
      pointerEvents="none"
      style={[
        styles.overlayBox,
        { left, top, width, height, borderColor },
      ]}
    >
      <View style={[styles.overlayLabel, isOcrFallback && styles.overlayLabelOcr]}>
        <Text style={[styles.overlayFormat, isOcrFallback && styles.overlayFormatOcr]}>
          {isOcrFallback ? `OCR \u2192 ${format}` : format}
        </Text>
        <Text style={styles.overlayText} numberOfLines={2}>{text}</Text>
        <Text style={styles.overlayConf}>{(confidence * 100).toFixed(0)}%</Text>
      </View>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// DEV-ONLY: Debug panel showing the exact image regions passed to ZXing
// ─────────────────────────────────────────────────────────────────────────────

function ZXingDebugPanel({ detections }: { detections: BarcodeDetection[] }) {
  // Show all detections including sentinels so logs always surface
  const debugDets = detections.filter(
    (d) => d.debugCropBase64 || (d.debugLogs && d.debugLogs.length > 0),
  );
  if (debugDets.length === 0) return null;

  // Collect all unique logs from first detection that has them
  const logs = debugDets.find((d) => d.debugLogs?.length)?.debugLogs ?? [];
  // Filter out sentinel-only entries that have no crop
  const cropDets = debugDets.filter((d) => d.debugCropBase64 && d.format !== '__debug__');

  return (
    <View pointerEvents="none" style={styles.debugPanel}>
      <Text style={styles.debugPanelTitle}>
        ZXing Debug (dev) — exact image passed to ZXing
      </Text>

      {/* Crop thumbnails */}
      {cropDets.length > 0 && (
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginBottom: 6 }}>
          {cropDets.map((det, idx) => (
            <View key={idx} style={styles.debugCropItem}>
              <Image
                source={{ uri: `data:image/jpeg;base64,${det.debugCropBase64}` }}
                style={styles.debugCropImage}
                resizeMode="contain"
              />
              <Text style={styles.debugCropFormat} numberOfLines={1}>
                {det.format || '?'}
              </Text>
              <Text style={styles.debugCropText} numberOfLines={1}>
                {det.text ? det.text.substring(0, 24) : '(no decode)'}
              </Text>
            </View>
          ))}
        </ScrollView>
      )}

      {/* Native pipeline log */}
      {logs.length > 0 && (
        <ScrollView style={styles.debugLogScroll} nestedScrollEnabled>
          {logs.map((line, i) => (
            <Text key={i} style={[
              styles.debugLogLine,
              line.includes('[ERROR]') || line.includes('THROW') ? styles.debugLogError :
              line.includes('VALID#') ? styles.debugLogSuccess :
              line.includes('[LUMA') || line.includes('[ROTATION') ? styles.debugLogHighlight :
              null,
            ]}>
              {line}
            </Text>
          ))}
        </ScrollView>
      )}
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Scan region overlay — transparent cutout in the center, dimmed surround
// ─────────────────────────────────────────────────────────────────────────────

function ScanRegionOverlay({
  containerWidth,
  containerHeight,
}: {
  containerWidth: number;
  containerHeight: number;
}) {
  // Square scan region — side = SCAN_REGION_FRAC * min(w, h)
  const side = Math.min(containerWidth, containerHeight) * SCAN_REGION_FRAC;
  const w = side;
  const h = side;
  const x = (containerWidth - w) / 2;
  const y = (containerHeight - h) / 2;
  const cornerLen = 20;
  const cornerW = 3;

  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      {/* Dimmed surround — four rectangles around the cutout */}
      <View style={[styles.dimOverlay, { top: 0, left: 0, right: 0, height: y }]} />
      <View style={[styles.dimOverlay, { top: y + h, left: 0, right: 0, bottom: 0 }]} />
      <View style={[styles.dimOverlay, { top: y, left: 0, width: x, height: h }]} />
      <View style={[styles.dimOverlay, { top: y, left: x + w, right: 0, height: h }]} />

      {/* Corner brackets */}
      {/* Top-left */}
      <View style={[styles.corner, { top: y, left: x, width: cornerLen, height: cornerW }]} />
      <View style={[styles.corner, { top: y, left: x, width: cornerW, height: cornerLen }]} />
      {/* Top-right */}
      <View style={[styles.corner, { top: y, right: containerWidth - x - w, width: cornerLen, height: cornerW }]} />
      <View style={[styles.corner, { top: y, right: containerWidth - x - w, width: cornerW, height: cornerLen }]} />
      {/* Bottom-left */}
      <View style={[styles.corner, { top: y + h - cornerW, left: x, width: cornerLen, height: cornerW }]} />
      <View style={[styles.corner, { top: y + h - cornerLen, left: x, width: cornerW, height: cornerLen }]} />
      {/* Bottom-right */}
      <View style={[styles.corner, { top: y + h - cornerW, right: containerWidth - x - w, width: cornerLen, height: cornerW }]} />
      <View style={[styles.corner, { top: y + h - cornerLen, right: containerWidth - x - w, width: cornerW, height: cornerLen }]} />

      {/* Label */}
      <Text style={[styles.scanRegionHint, { top: y + h + 14, alignSelf: 'center' }]}>
        Position barcode inside the box
      </Text>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Permission gate
// ─────────────────────────────────────────────────────────────────────────────

function PermissionGate() {
  return (
    <View style={styles.permissionContainer}>
      <Text style={styles.permissionText}>
        Camera permission is required for barcode scanning.
      </Text>
      <TouchableOpacity
        style={styles.permissionButton}
        onPress={() => Linking.openSettings()}
      >
        <Text style={styles.permissionButtonText}>Open Settings</Text>
      </TouchableOpacity>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────

export default function BarcodeScanner({
  onBarcodeDetected,
  detectionOptions = { confidence: 0.35, modelInputSize: 640, maxDetections: 10 },
  cooldownMs = 2000,
  children,
  style,
  facing = 'back',
}: BarcodeScannerProps) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device: CameraDevice | undefined = useCameraDevice(facing);

  // ── Camera format: pick 1080p for high-res ZXing crops ────────────────────
  const format = useCameraFormat(device, [
    { videoResolution: { width: 1920, height: 1080 } },
    { fps: 30 },
  ]);

  // ── Torch state ───────────────────────────────────────────────────────────
  // 'on' = always on, 'off' = always off, 'auto' = managed by native brightness
  const torchMode = detectionOptions.torchMode ?? 'off';
  const [autoTorchOn, setAutoTorchOn] = useState(false);
  const torchEnabled = torchMode === 'on' || (torchMode === 'auto' && autoTorchOn);

  // ── Zoom & focus ──────────────────────────────────────────────────────────
  const cameraRef = useRef<Camera>(null);
  const [zoom, setZoom] = useState(1);
  const savedZoom = useSharedValue(1);
  const minZoom = device?.minZoom ?? 1;
  const maxZoom = Math.min(device?.maxZoom ?? 16, 16);

  const pinchGesture = Gesture.Pinch()
    .onStart(() => { savedZoom.value = zoom; })
    .onUpdate((e) => {
      const next = Math.max(minZoom, Math.min(savedZoom.value * e.scale, maxZoom));
      runOnJS(setZoom)(next);
    });

  const focusOnPoint = useCallback((x: number, y: number) => {
    cameraRef.current?.focus({ x, y }).catch(() => {});
  }, []);

  const tapGesture = Gesture.Tap()
    .maxDuration(300)
    .onEnd((e) => { runOnJS(focusOnPoint)(e.x, e.y); });

  const combinedGesture = Gesture.Race(pinchGesture, tapGesture);

  // ── Detection ─────────────────────────────────────────────────────────────
  const { detections, frameProcessor, frameWidth: sensorW, frameHeight: sensorH } = useZXingNanoDet(
    __DEV__ ? { ...detectionOptions, debug: true } : detectionOptions,
  );

  // Container dimensions (for overlay scaling)
  const [containerSize, setContainerSize] = useState({ width: 1, height: 1 });

  const frameSize = { width: sensorW || 640, height: sensorH || 480 };

  // ── Auto-torch: monitor average frame brightness via native logs ──────────
  // The native plugin reports frame brightness in debug logs. For auto mode we
  // approximate by checking if most detections are low-confidence (heuristic).
  // A proper implementation would expose avgLuma from native side; for now we
  // toggle torch when there are zero detections for several consecutive frames.
  const darkFrameCountRef = useRef(0);
  const brightFrameCountRef = useRef(0);
  const AUTO_DARK_FRAMES = 10;  // frames with 0 detects before enabling torch
  const AUTO_BRIGHT_FRAMES = 5; // frames with detects before disabling torch

  useEffect(() => {
    if (torchMode !== 'auto') {
      setAutoTorchOn(false);
      darkFrameCountRef.current = 0;
      brightFrameCountRef.current = 0;
      return;
    }
    if (detections.length === 0) {
      darkFrameCountRef.current += 1;
      brightFrameCountRef.current = 0;
      if (darkFrameCountRef.current >= AUTO_DARK_FRAMES) setAutoTorchOn(true);
    } else {
      brightFrameCountRef.current += 1;
      darkFrameCountRef.current = 0;
      if (brightFrameCountRef.current >= AUTO_BRIGHT_FRAMES) setAutoTorchOn(false);
    }
  }, [detections, torchMode]);

  // ── Scan-region filter ────────────────────────────────────────────────────
  // Compute the scan region in **sensor** (landscape 640×480) coordinates.
  // The display is portrait; the sensor is landscape → 90° CW rotation.
  // Display center fractional rect → sensor fractional rect (axes swapped).
  const filteredDetections = useMemo(() => {
    const cw = containerSize.width;
    const ch = containerSize.height;
    if (cw <= 1 || ch <= 1) return detections; // not laid-out yet
    const fw = frameSize.width;
    const fh = frameSize.height;
    const needsRotation = fw > fh && ch > cw;

    // Scan region in display coords (centered square)
    const side = Math.min(cw, ch) * SCAN_REGION_FRAC;
    const dispRegX = (cw - side) / 2;
    const dispRegY = (ch - side) / 2;
    const dispRegW = side;
    const dispRegH = side;

    return detections.filter((det) => {
      const bb = det.boundingBox;
      // Convert bounding box to display coords
      let dx: number, dy: number, dw: number, dh: number;
      if (needsRotation) {
        dx = (fh - bb.y - bb.height) * (cw / fh);
        dy = bb.x * (ch / fw);
        dw = bb.height * (cw / fh);
        dh = bb.width * (ch / fw);
      } else {
        dx = bb.x * (cw / fw);
        dy = bb.y * (ch / fh);
        dw = bb.width * (cw / fw);
        dh = bb.height * (ch / fh);
      }
      // Check center of detection falls within the scan region
      const centerX = dx + dw / 2;
      const centerY = dy + dh / 2;
      return (
        centerX >= dispRegX && centerX <= dispRegX + dispRegW &&
        centerY >= dispRegY && centerY <= dispRegY + dispRegH
      );
    });
  }, [detections, containerSize, frameSize]);

  // ── Consensus algorithm ───────────────────────────────────────────────────
  // Buffer recent reads; only report a barcode once N identical values seen.
  const consensusEnabled = detectionOptions.enableConsensus ?? false;
  const consensusRequired = detectionOptions.consensusCount ?? 3;
  const consensusBufferRef = useRef<Map<string, number>>(new Map());

  const confirmedDetections = useMemo(() => {
    if (!consensusEnabled) return filteredDetections;

    const buf = consensusBufferRef.current;
    const confirmed: BarcodeDetection[] = [];

    for (const det of filteredDetections) {
      if (!det.text || det.format === 'UNKNOWN' || det.format === '__debug__') continue;
      const key = `${det.format}:${det.text}`;
      const count = (buf.get(key) ?? 0) + 1;
      buf.set(key, count);
      if (count >= consensusRequired) {
        confirmed.push({ ...det, consensusCount: count, source: 'consensus' });
        buf.delete(key); // reset after confirming
      }
    }

    // Decay stale entries (keys not seen this frame drop by 1)
    const currentKeys = new Set(filteredDetections.map((d) => `${d.format}:${d.text}`));
    for (const [k, v] of buf.entries()) {
      if (!currentKeys.has(k)) {
        if (v <= 1) buf.delete(k);
        else buf.set(k, v - 1);
      }
    }

    return confirmed;
  }, [filteredDetections, consensusEnabled, consensusRequired]);

  // ── Choose which detections to surface ────────────────────────────────────
  const surfacedDetections = consensusEnabled ? confirmedDetections : filteredDetections;

  // ── Conditional bbox display ──────────────────────────────────────────────
  // Only show bounding box overlays when:
  //  • ZXing decoded the barcode (format not UNKNOWN), OR
  //  • OCR fallback produced text, OR
  //  • Direct-ZXing pass decoded something
  const visibleOverlays = useMemo(
    () => surfacedDetections.filter(
      (det) => det.text && det.text.length > 0 && det.format !== 'UNKNOWN' && det.format !== '__debug__',
    ),
    [surfacedDetections],
  );

  // Deduplicate fired callbacks via a cooldown map
  const cooldownMap = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  const handleDetection = useCallback(
    (det: BarcodeDetection) => {
      if (!onBarcodeDetected) return;
      const now = Date.now();
      const key = `${det.format}:${det.text}`;
      const last = cooldownMap.current.get(key) ?? 0;
      if (now - last < cooldownMs) return;
      cooldownMap.current.set(key, now);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      onBarcodeDetected(det);
    },
    [onBarcodeDetected, cooldownMs],
  );

  useEffect(() => {
    surfacedDetections.forEach(handleDetection);
  }, [surfacedDetections, handleDetection]);

  if (!hasPermission) return <PermissionGate />;
  if (!device) {
    return (
      <View style={[styles.fill, styles.centered, style]}>
        <Text style={styles.permissionText}>No camera available.</Text>
      </View>
    );
  }

  return (
    <GestureHandlerRootView style={[styles.fill, style]}>
      <GestureDetector gesture={combinedGesture}>
        <View
          style={styles.fill}
          onLayout={(e) => {
            const { width, height } = e.nativeEvent.layout;
            setContainerSize({ width, height });
          }}
        >
          {/* Camera — explicit 1080p format for high-res ZXing crops */}
          <Camera
            ref={cameraRef}
            style={StyleSheet.absoluteFill}
            device={device}
            isActive
            photo
            frameProcessor={frameProcessor}
            pixelFormat="yuv"
            zoom={zoom}
            torch={torchEnabled ? 'on' : 'off'}
            {...(format ? { format } : {})}
          />

          {/* Scan region overlay */}
          <ScanRegionOverlay
            containerWidth={containerSize.width}
            containerHeight={containerSize.height}
          />

          {/* Zoom indicator */}
          {zoom > 1.05 && (
            <View pointerEvents="none" style={styles.zoomBadge}>
              <Text style={styles.zoomBadgeText}>{zoom.toFixed(1)}x</Text>
            </View>
          )}

          {/* Bounding-box overlays — only shown when barcode is decoded */}
          {visibleOverlays.map((det: BarcodeDetection, idx: number) => (
            <OverlayBox
              key={`${det.format}-${det.text}-${idx}`}
              detection={det}
              frameWidth={frameSize.width}
              frameHeight={frameSize.height}
              containerWidth={containerSize.width}
              containerHeight={containerSize.height}
            />
          ))}

          {/* Caller-provided children */}
          {children}

          {/* DEV-ONLY: ZXing input crop viewer */}
          {__DEV__ && <ZXingDebugPanel detections={surfacedDetections} />}
        </View>
      </GestureDetector>
    </GestureHandlerRootView>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────────────────────────────────────

const BORDER_COLOR     = '#00E5FF';
const OCR_BORDER_COLOR = '#FFB300'; // amber — indicates PaddleOCR fallback
const LABEL_BG         = 'rgba(0, 0, 0, 0.65)';
const LABEL_BG_OCR     = 'rgba(80, 50, 0, 0.75)';

const styles = StyleSheet.create({
  fill:    { flex: 1 },
  centered:{ alignItems: 'center', justifyContent: 'center' },

  // ── Overlay box ───────────────────────────────────────────────────────────
  overlayBox: {
    position:     'absolute',
    borderWidth:  2,
    borderColor:  BORDER_COLOR,
    borderRadius: 4,
    overflow:     'visible',
  },
  overlayLabel: {
    position:        'absolute',
    bottom:          -52,
    left:            0,
    right:           0,
    backgroundColor: LABEL_BG,
    paddingHorizontal: 6,
    paddingVertical:   4,
    borderRadius:    4,
    alignItems:      'center',
  },
  overlayLabelOcr: {
    backgroundColor: LABEL_BG_OCR,
  },
  overlayFormat: {
    color:      BORDER_COLOR,
    fontSize:   10,
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  overlayFormatOcr: {
    color: OCR_BORDER_COLOR,
  },
  overlayText: {
    color:    '#FFFFFF',
    fontSize: 13,
    fontWeight: '500',
    textAlign: 'center',
  },
  overlayConf: {
    color:    'rgba(255,255,255,0.55)',
    fontSize: 10,
  },

  // ── Scan region ────────────────────────────────────────────────────────────
  dimOverlay: {
    position:        'absolute',
    backgroundColor: 'rgba(0, 0, 0, 0.55)',
  },
  corner: {
    position:        'absolute',
    backgroundColor: '#00E5FF',
    borderRadius:    1,
  },
  scanRegionHint: {
    position:   'absolute',
    left:       0,
    right:      0,
    textAlign:  'center',
    color:      'rgba(255,255,255,0.7)',
    fontSize:   13,
  },

  // ── Permission gate ───────────────────────────────────────────────────────
  permissionContainer: {
    flex:            1,
    alignItems:      'center',
    justifyContent:  'center',
    padding:         24,
    backgroundColor: '#000',
  },
  permissionText: {
    color:     '#FFF',
    fontSize:  16,
    textAlign: 'center',
    marginBottom: 16,
  },
  permissionButton: {
    backgroundColor: '#00E5FF',
    paddingHorizontal: 24,
    paddingVertical:   12,
    borderRadius: 8,
  },
  permissionButtonText: {
    color:      '#000',
    fontWeight: '600',
    fontSize:   15,
  },

  // ── DEV debug panel ───────────────────────────────────────────────────────
  debugPanel: {
    position:         'absolute',
    bottom:           0,
    left:             0,
    right:            0,
    maxHeight:        260,
    backgroundColor:  'rgba(0, 0, 0, 0.88)',
    paddingVertical:  8,
    paddingHorizontal: 10,
  },
  debugPanelTitle: {
    color:        '#FFD600',
    fontSize:     10,
    fontWeight:   '700',
    letterSpacing: 0.8,
    marginBottom:  6,
    textTransform: 'uppercase',
  },
  debugCropItem: {
    marginRight:   10,
    alignItems:    'center',
  },
  debugCropImage: {
    width:        100,
    height:       60,
    borderWidth:  1,
    borderColor:  '#FFD600',
    borderRadius: 3,
    backgroundColor: '#111',
  },
  debugCropFormat: {
    color:    '#FFD600',
    fontSize: 9,
    marginTop: 3,
  },
  debugCropText: {
    color:    '#FFF',
    fontSize: 9,
  },
  debugLogScroll: {
    maxHeight: 130,
  },
  debugLogLine: {
    color:        'rgba(255,255,255,0.75)',
    fontSize:     8.5,
    fontFamily:   Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    lineHeight:   13,
  },
  debugLogError: {
    color: '#FF5252',
  },
  debugLogSuccess: {
    color: '#69FF47',
  },
  debugLogHighlight: {
    color: '#40C4FF',
  },

  // ── Zoom indicator ────────────────────────────────────────────────────────
  zoomBadge: {
    position:         'absolute',
    top:              12,
    alignSelf:        'center',
    backgroundColor:  'rgba(0,0,0,0.55)',
    paddingHorizontal: 12,
    paddingVertical:   4,
    borderRadius:     14,
  },
  zoomBadgeText: {
    color:      '#FFFFFF',
    fontSize:   13,
    fontWeight: '600',
  },
});
