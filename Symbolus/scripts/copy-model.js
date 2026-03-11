#!/usr/bin/env node
/**
 * scripts/copy-model.js
 *
 * Copies the NanoDet ONNX model from the root of the repo into the Symbolus
 * assets folder so it gets bundled with the app.
 *
 * Run once after cloning:
 *   node scripts/copy-model.js
 *
 * Or add to package.json scripts:
 *   "postinstall": "node scripts/copy-model.js"
 */

const fs   = require('fs');
const path = require('path');

const SRC  = path.resolve(__dirname, '../../nanodet_barcode.onnx');
// Try the 416-specific variant first (smaller / faster)
const SRC2 = path.resolve(__dirname, '../../nanodet_barcode_416.onnx');
const DEST_DIR = path.resolve(__dirname, '../assets/models');
const DEST     = path.join(DEST_DIR, 'nanodet_barcode.onnx');

// Also copy to Android assets for the module
const ANDROID_ASSETS = path.resolve(
  __dirname,
  '../modules/zxing-nanodet/android/src/main/assets',
);
const ANDROID_DEST = path.join(ANDROID_ASSETS, 'nanodet_barcode.onnx');

function copyIfNeeded(src, dest) {
  const dir = path.dirname(dest);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  if (!fs.existsSync(dest)) {
    fs.copyFileSync(src, dest);
    console.log(`✓ Copied model → ${path.relative(process.cwd(), dest)}`);
  } else {
    console.log(`  Model already present at ${path.relative(process.cwd(), dest)}`);
  }
}

const modelSrc = fs.existsSync(SRC2) ? SRC2 : SRC;

if (!fs.existsSync(modelSrc)) {
  console.warn(
    `[copy-model] NanoDet model not found (non-fatal).\n` +
    `  Expected one of:\n  ${SRC2}\n  ${SRC}\n` +
    `  Run the Python export script first, or copy the model manually.`,
  );
  process.exit(0); // non-fatal — app will build without barcode detection
}

copyIfNeeded(modelSrc, DEST);
copyIfNeeded(modelSrc, ANDROID_DEST);
console.log('Done.');
