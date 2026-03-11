// SPDX-License-Identifier: Apache-2.0
// BarcodeScanner.tsx
//
// Full-screen camera view with:
//  • Live NanoDet bounding-box overlays (one Animated.View per detection)
//  • ZXing-decoded text label beneath each box
//  • Haptic + callback when a new barcode is confirmed

import React, {
  useCallback,
  useEffect,
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
  useCameraPermission,
  type CameraDevice,
} from 'react-native-vision-camera';
import * as Haptics from 'expo-haptics';

import {
  useZXingNanoDet,
  type BarcodeDetection,
  type DetectBarcodesOptions,
} from '../modules/zxing-nanodet/src';

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
  const { boundingBox, text, format, confidence } = detection;

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
        { left, top, width, height },
      ]}
    >
      <View style={styles.overlayLabel}>
        <Text style={styles.overlayFormat}>{format}</Text>
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

  const { detections, frameProcessor, frameWidth: sensorW, frameHeight: sensorH } = useZXingNanoDet(
    __DEV__ ? { ...detectionOptions, debug: true } : detectionOptions,
  );

  console.log(
    `[BarcodeScanner] render — detections.length=${detections.length}`,
    detections.length > 0 ? JSON.stringify(detections[0]?.boundingBox) : '',
  );

  // Container dimensions (for overlay scaling)
  const [containerSize, setContainerSize] = useState({ width: 1, height: 1 });

  // Use actual frame dimensions from the frame processor hook.
  // Sensor delivers landscape frames (e.g. 640×480); we need to handle rotation.
  const frameSize = { width: sensorW || 640, height: sensorH || 480 };

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

  // Fire callback for each unique new detection
  useEffect(() => {
    detections.forEach(handleDetection);
  }, [detections, handleDetection]);

  if (!hasPermission) return <PermissionGate />;
  if (!device) {
    return (
      <View style={[styles.fill, styles.centered, style]}>
        <Text style={styles.permissionText}>No camera available.</Text>
      </View>
    );
  }

  return (
    <View
      style={[styles.fill, style]}
      onLayout={(e) => {
        const { width, height } = e.nativeEvent.layout;
        console.log(`[BarcodeScanner] onLayout — container: ${width}x${height}`);
        setContainerSize({ width, height });
      }}
    >
      {/* Camera */}
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
        onOutputOrientationChanged={() => {
          // Frame dimensions are now tracked via the hook's frameWidth/frameHeight
          console.log('[BarcodeScanner] onOutputOrientationChanged');
        }}
      />

      {/* Bounding-box overlays */}
      {(() => {
        console.log(
          `[BarcodeScanner] rendering overlays — count=${detections.length}` +
          ` frame=${frameSize.width}x${frameSize.height}` +
          ` container=${containerSize.width}x${containerSize.height}`,
        );
        return null;
      })()}
      {(detections as BarcodeDetection[]).map((det: BarcodeDetection, idx: number) => (
        <OverlayBox
          key={`${det.format}-${det.text}-${idx}`}
          detection={det}
          frameWidth={frameSize.width}
          frameHeight={frameSize.height}
          containerWidth={containerSize.width}
          containerHeight={containerSize.height}
        />
      ))}

      {/* Scanning reticle hint */}
      {detections.length === 0 && (
        <View pointerEvents="none" style={styles.reticleContainer}>
          <View style={styles.reticle} />
          <Text style={styles.reticleHint}>Point at a barcode</Text>
        </View>
      )}

      {/* Caller-provided children (e.g. close button) */}
      {children}

      {/* ── DEV-ONLY: ZXing input crop viewer ───────────────────────────── */}
      {__DEV__ && <ZXingDebugPanel detections={detections} />}
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────────────────────────────────────

const BORDER_COLOR = '#00E5FF';
const LABEL_BG     = 'rgba(0, 0, 0, 0.65)';

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
  overlayFormat: {
    color:      BORDER_COLOR,
    fontSize:   10,
    fontWeight: '600',
    letterSpacing: 0.5,
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

  // ── Reticle ───────────────────────────────────────────────────────────────
  reticleContainer: {
    ...StyleSheet.absoluteFillObject,
    alignItems:      'center',
    justifyContent:  'center',
    pointerEvents:   'none',
  },
  reticle: {
    width:        220,
    height:       140,
    borderWidth:  2,
    borderColor:  'rgba(255,255,255,0.5)',
    borderRadius: 10,
  },
  reticleHint: {
    marginTop:  12,
    color:      'rgba(255,255,255,0.7)',
    fontSize:   14,
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
});
