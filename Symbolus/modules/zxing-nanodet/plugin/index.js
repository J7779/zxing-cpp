// SPDX-License-Identifier: Apache-2.0
// Expo config plugin for zxing-nanodet.
//
// Applied during `expo prebuild` to:
//   1. Add the module to android/settings.gradle
//   2. Add `implementation project(':zxing-nanodet')` to android/app/build.gradle
//   3. Insert ZXingNanoDetPluginPackage.init() call into MainApplication.kt

const { withSettingsGradle, withAppBuildGradle, withDangerousMod } = require('@expo/config-plugins');
const path  = require('path');
const fs    = require('fs');

// ─────────────────────────────────────────────────────────────────────────────
// settings.gradle patch
// ─────────────────────────────────────────────────────────────────────────────
function patchSettings(contents) {
  if (contents.includes(':zxing-nanodet')) return contents;
  return (
    contents.trimEnd() +
    `\n\n// ─── zxing-nanodet (NanoDet + ZXing frame processor plugin) ─────────\n` +
    `include ':zxing-nanodet'\n` +
    `project(':zxing-nanodet').projectDir = ` +
    `new File(rootProject.projectDir, '../modules/zxing-nanodet/android')\n`
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// app/build.gradle patch
// ─────────────────────────────────────────────────────────────────────────────
function patchAppBuildGradle(contents) {
  if (contents.includes(':zxing-nanodet')) return contents;
  // Insert after the opening `dependencies {` line
  return contents.replace(
    /^(dependencies\s*\{)/m,
    `$1\n    implementation project(':zxing-nanodet')`
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MainApplication.kt patch
// ─────────────────────────────────────────────────────────────────────────────
function patchMainApplication(contents) {
  if (contents.includes('ZXingNanoDetPluginPackage')) return contents;

  // Add import — insert after the last existing import line
  contents = contents.replace(
    /(import\s+[\w.]+\n)(\n*class\s)/,
    `$1import expo.modules.zxing.nanodet.ZXingNanoDetPluginPackage\n$2`
  );

  // Add init call — right after super.onCreate()
  contents = contents.replace(
    /(super\.onCreate\(\))/,
    `$1\n    ZXingNanoDetPluginPackage.init()`
  );

  return contents;
}

// ─────────────────────────────────────────────────────────────────────────────
// Find MainApplication.kt anywhere in the java/ source tree
// ─────────────────────────────────────────────────────────────────────────────
function findMainApplication(dir) {
  if (!fs.existsSync(dir)) return null;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      const found = findMainApplication(full);
      if (found) return found;
    } else if (entry.name === 'MainApplication.kt') {
      return full;
    }
  }
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Plugin entry point
// ─────────────────────────────────────────────────────────────────────────────
module.exports = function withZXingNanoDet(config) {
  // 1. settings.gradle
  config = withSettingsGradle(config, (c) => {
    c.modResults.contents = patchSettings(c.modResults.contents);
    return c;
  });

  // 2. app/build.gradle
  config = withAppBuildGradle(config, (c) => {
    c.modResults.contents = patchAppBuildGradle(c.modResults.contents);
    return c;
  });

  // 3. MainApplication.kt
  config = withDangerousMod(config, [
    'android',
    (c) => {
      const javaSrc = path.join(
        c.modRequest.platformProjectRoot,
        'app', 'src', 'main', 'java'
      );
      const mainAppPath = findMainApplication(javaSrc);
      if (mainAppPath) {
        const before = fs.readFileSync(mainAppPath, 'utf8');
        const after  = patchMainApplication(before);
        if (after !== before) {
          fs.writeFileSync(mainAppPath, after, 'utf8');
          console.log('[zxing-nanodet] Patched', path.relative(process.cwd(), mainAppPath));
        }
      } else {
        console.warn('[zxing-nanodet] Could not find MainApplication.kt — plugin registration skipped.');
      }
      return c;
    },
  ]);

  return config;
};
