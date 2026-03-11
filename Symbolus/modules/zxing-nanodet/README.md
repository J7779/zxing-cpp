# zxing-nanodet

React Native Vision Camera (v4) **Frame Processor Plugin** that combines:

- **NanoDet-Plus** ONNX barcode *detection* (bounding box localisation)
- **ZXing-C++** barcode *decoding* (format + text)

running entirely in native code (iOS Objective-C++ / Android Kotlin+JNI) for
maximum performance on every camera frame.

---

## Architecture

```
Camera frame  (VisionCamera)
      │
      ▼
detectBarcodes(frame, options)   ← JS worklet call
      │
      ▼  (native, off JS thread)
┌─────────────────────────────────────┐
│  1. Pixel buffer → RGBA             │
│  2. NanoDet ORT inference           │  ← onnxruntime-c / onnxruntime-android
│     Preprocess (letterbox + norm)   │
│     Run model (416 × 416)           │
│     PostprocessGFL + NMS            │
│  3. For each detection:             │
│     a. Crop luma ROI (+ 10% pad)    │
│     b. ZXing ReadBarcodes(crop)     │  ← ZXing-CPP core (C++)
│  4. Return [{format,text,bbox,…}]   │
└─────────────────────────────────────┘
      │
      ▼
runOnJS(setDetections)(results)    ← React state update
      │
      ▼
<BarcodeScanner>  renders overlay boxes
```

---

## File layout

```
modules/zxing-nanodet/
├── package.json
└── src/
    ├── index.ts              JS entry point
    ├── types.ts              TypeScript types
    ├── frameProcessor.ts     createFrameProcessorPlugin wrapper
    └── useZXingNanoDet.ts    React hook
ios/
├── ZXingNanoDetPlugin.h      ObjC++ header
├── ZXingNanoDetPlugin.mm     iOS frame processor plugin (ORT C API + ZXing C++)
├── ZXingNanoDetORT.h         Helper header
└── zxing-nanodet.podspec     CocoaPods spec
android/
├── build.gradle
└── src/main/
    ├── java/expo/modules/zxing/nanodet/
    │   ├── ZXingNanoDetPlugin.kt        VisionCamera plugin
    │   ├── ZXingNanoDetJNI.kt           JNI stub
    │   └── ZXingNanoDetPluginPackage.kt Registry entry
    ├── cpp/
    │   ├── CMakeLists.txt
    │   └── ZXingNanoDetJNI.cpp          JNI bridge (NanoDet + ZXing)
    └── assets/
        └── nanodet_barcode.onnx         (copied by scripts/copy-model.js)
```

---

## Setup

### 1. Copy the ONNX model

```bash
cd Symbolus
node scripts/copy-model.js
```

This copies `nanodet_barcode_416.onnx` (or `nanodet_barcode.onnx`) from the
repo root into:
- `Symbolus/assets/models/nanodet_barcode.onnx`  (iOS bundle resource)
- `Symbolus/modules/zxing-nanodet/android/src/main/assets/nanodet_barcode.onnx`

### 2. Install JS dependencies

```bash
cd Symbolus
npm install
```

### 3. Prebuild (generates iOS/Android project files)

```bash
npx expo prebuild --clean
```

### 4. iOS — install pods

```bash
cd ios && pod install && cd ..
```

In Xcode, add `nanodet_barcode.onnx` to the target's **Bundle Resources** if
not already included via Auto-Linking.

### 5. Run

```bash
npx expo run:ios
npx expo run:android
```

---

## JS usage

### Hook (recommended)

```tsx
import BarcodeScanner from '@/components/BarcodeScanner';

export default function ScanScreen() {
  return (
    <BarcodeScanner
      onBarcodeDetected={(det) =>
        console.log(det.format, det.text, det.boundingBox)
      }
    />
  );
}
```

### Manual frame processor

```tsx
import { Camera } from 'react-native-vision-camera';
import { useZXingNanoDet } from 'zxing-nanodet';

const { frameProcessor, detections } = useZXingNanoDet({ confidence: 0.35 });

<Camera device={device} isActive frameProcessor={frameProcessor} />;
```

---

## Model

Uses **NanoDet-Plus** trained on a custom barcode dataset at 416 × 416
resolution.  The model pipeline mirrors the WASM wrapper in
`wrappers/wasm/BarcodeReader.cpp` and uses the same C++ pre/post-processing
code from `core/src/onnx/NanoDet.h`.

| Model file              | Input  | Size  |
|-------------------------|--------|-------|
| nanodet_barcode_416.onnx | 416×416 | ~4.9 MB |

---

## License

Apache-2.0 — see [LICENSE](../../LICENSE).
