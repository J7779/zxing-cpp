// SPDX-License-Identifier: Apache-2.0
// ZXingNanoDetJNI.cpp
//
// Three JNI functions exposed to Kotlin:
//
//  1. nativePreprocess(rgba, w, h, targetSize)
//     â†’ float[] = [tensor floats..., scale, padX_f, padY_f, newW_f, newH_f]
//     Runs NanoDet::Preprocess() (letterbox + BGR normalize). No ORT.
//
//  2. nativePostprocessGFL(output, numBoxes, boxSize, srcW, srcH,
//                           scale, padX, padY, targetSize, confidence)
//     â†’ String[][] boxes: each String[] = [x1, y1, x2, y2, score, classId]
//     Decodes GFL output + NMS.
//
//  3. nativeDecodeBarcode(rgba, frameW, frameH, cropX, cropY, cropW, cropH)
//     â†’ String[][] results: each String[] =
//       [format, text, cx0, cy0, cx1, cy1, cx2, cy2, cx3, cy3]
//     Crops luma from RGBA frame, runs ZXing ReadBarcodes.
//
// ORT model inference runs entirely in Kotlin using the Android ORT Java API.

#include <jni.h>
#include <android/log.h>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// NanoDet preprocessing/postprocessing (no ZXING_USE_ONNXRUNTIME â€” no ORT headers)
#include "NanoDet.h"

// ZXing barcode decoding
#include "ReadBarcode.h"
#include "BarcodeFormat.h"
#include "ImageView.h"
#include "MultiFormatReader.h"

#define LOG_TAG "ZXingNanoDetJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace ZXing;

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// 1. nativePreprocess
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_expo_modules_zxing_nanodet_ZXingNanoDetJNI_nativePreprocess(
    JNIEnv* env, jclass /*cls*/,
    jbyteArray jRGBA, jint width, jint height, jint targetSize)
{
    const jsize rgbaLen = env->GetArrayLength(jRGBA);
    if (rgbaLen < width * height * 4) return env->NewFloatArray(0);

    jbyte* raw = env->GetByteArrayElements(jRGBA, nullptr);
    const uint8_t* rgba = reinterpret_cast<const uint8_t*>(raw);

    auto result = NanoDet::Preprocess(rgba, width, height, targetSize);

    env->ReleaseByteArrayElements(jRGBA, raw, JNI_ABORT);

    // Return tensor + 5 metadata floats at the end:
    // [0 .. tensorSize-1] = float tensor (CHW)
    // [tensorSize+0] = scale
    // [tensorSize+1] = padX (as float)
    // [tensorSize+2] = padY (as float)
    // [tensorSize+3] = newWidth (as float)
    // [tensorSize+4] = newHeight (as float)
    const jsize tensorSize = (jsize)result.tensor.size();
    jfloatArray out = env->NewFloatArray(tensorSize + 5);
    if (!out) return env->NewFloatArray(0);

    env->SetFloatArrayRegion(out, 0, tensorSize, result.tensor.data());
    float meta[5] = {
        result.scale,
        (float)result.padX,
        (float)result.padY,
        (float)result.newWidth,
        (float)result.newHeight,
    };
    env->SetFloatArrayRegion(out, tensorSize, 5, meta);
    return out;
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// 2. nativePostprocessGFL
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_expo_modules_zxing_nanodet_ZXingNanoDetJNI_nativePostprocessGFL(
    JNIEnv* env, jclass /*cls*/,
    jfloatArray jOutput,
    jint numBoxes, jint boxSize,
    jint srcW, jint srcH,
    jfloat scale, jfloat padX, jfloat padY,
    jint targetSize, jfloat confidence)
{
    jclass stringClass = env->FindClass("java/lang/String");
    jclass stringArrayClass = env->FindClass("[Ljava/lang/String;");

    const jsize outLen = env->GetArrayLength(jOutput);
    jfloat* outData = env->GetFloatArrayElements(jOutput, nullptr);

    // Auto-detect output format (matches WASM logic)
    bool isDecoded = (boxSize == 5 || boxSize == 6);

    std::vector<NanoDet::Detection> dets;
    if (isDecoded) {
        dets = NanoDet::PostprocessDecoded(
            outData, numBoxes, boxSize,
            srcW, srcH,
            scale, (int)padX, (int)padY,
            confidence);
    } else {
        dets = NanoDet::PostprocessGFL(
            outData, numBoxes, boxSize,
            srcW, srcH,
            scale, (int)padX, (int)padY,
            targetSize, confidence);
    }
    env->ReleaseFloatArrayElements(jOutput, outData, JNI_ABORT);

    dets = NanoDet::ApplyNMS(dets, 0.45f);

    jobjectArray outer = env->NewObjectArray((jsize)dets.size(), stringArrayClass, nullptr);
    for (int i = 0; i < (int)dets.size(); ++i) {
        const auto& d = dets[i];
        // [x1, y1, x2, y2, score, classId]
        jobjectArray inner = env->NewObjectArray(6, stringClass, nullptr);
        auto set = [&](int idx, const std::string& s) {
            env->SetObjectArrayElement(inner, idx, env->NewStringUTF(s.c_str()));
        };
        set(0, std::to_string(d.x1));
        set(1, std::to_string(d.y1));
        set(2, std::to_string(d.x2));
        set(3, std::to_string(d.y2));
        set(4, std::to_string(d.score));
        set(5, std::to_string(d.classId));
        env->SetObjectArrayElement(outer, i, inner);
        env->DeleteLocalRef(inner);
    }
    return outer;
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// 3. nativeDecodeBarcode
// Crops luma from RGBA frame, optionally rotates 90° CW, runs ZXing.
// When debug=true, prepends a ["__log__", logText] entry to the result.
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_expo_modules_zxing_nanodet_ZXingNanoDetJNI_nativeDecodeBarcode(
    JNIEnv* env, jclass /*cls*/,
    jbyteArray jRGBA,
    jint frameW, jint frameH,
    jint cropX, jint cropY, jint cropW, jint cropH,
    jboolean jDebug)
{
    bool dbg = (jDebug == JNI_TRUE);
    jclass stringClass = env->FindClass("java/lang/String");
    jclass stringArrayClass = env->FindClass("[Ljava/lang/String;");

    // Accumulate debug log lines (only used when dbg=true)
    std::vector<std::string> logLines;
    auto addLog = [&](const std::string& line) {
        LOGI("%s", line.c_str());
        if (dbg) logLines.push_back(line);
    };

    auto makeEmpty = [&]() {
        if (!dbg || logLines.empty()) return env->NewObjectArray(0, stringArrayClass, nullptr);
        // Return log entry even on early-exit
        std::string joined;
        for (auto& l : logLines) { joined += l; joined += "\n"; }
        jobjectArray outer = env->NewObjectArray(1, stringArrayClass, nullptr);
        jobjectArray logRow = env->NewObjectArray(2, stringClass, nullptr);
        env->SetObjectArrayElement(logRow, 0, env->NewStringUTF("__log__"));
        env->SetObjectArrayElement(logRow, 1, env->NewStringUTF(joined.c_str()));
        env->SetObjectArrayElement(outer, 0, logRow);
        env->DeleteLocalRef(logRow);
        return outer;
    };

    addLog("[FRAME] " + std::to_string(frameW) + "x" + std::to_string(frameH) +
           " landscape=" + (frameW > frameH ? "true" : "false"));
    addLog("[CROP_IN] x=" + std::to_string(cropX) + " y=" + std::to_string(cropY) +
           " w=" + std::to_string(cropW) + " h=" + std::to_string(cropH));

    if (cropW <= 0 || cropH <= 0) {
        addLog("[ERROR] crop dimensions <=0, aborting");
        return makeEmpty();
    }

    const jsize rgbaLen = env->GetArrayLength(jRGBA);
    if (rgbaLen < frameW * frameH * 4) {
        addLog("[ERROR] RGBA buffer too small: got " + std::to_string(rgbaLen) +
               " need " + std::to_string(frameW * frameH * 4));
        return makeEmpty();
    }

    jbyte* raw = env->GetByteArrayElements(jRGBA, nullptr);
    const uint8_t* rgba = reinterpret_cast<const uint8_t*>(raw);

    // Clamp crop to frame bounds
    int bx = std::max(0, cropX);
    int by = std::max(0, cropY);
    int bw = std::min(frameW - bx, cropW);
    int bh = std::min(frameH - by, cropH);
    addLog("[CROP_CLAMPED] x=" + std::to_string(bx) + " y=" + std::to_string(by) +
           " w=" + std::to_string(bw) + " h=" + std::to_string(bh));

    // Compute luma stats from RGBA for diagnostics
    long lumaSum = 0;
    uint8_t lumaMin = 255, lumaMax = 0;
    for (int row = 0; row < bh; ++row) {
        for (int col = 0; col < bw; ++col) {
            int srcIdx = ((by + row) * frameW + (bx + col)) * 4;
            int r = rgba[srcIdx + 0] & 0xFF;
            int g = rgba[srcIdx + 1] & 0xFF;
            int b = rgba[srcIdx + 2] & 0xFF;
            uint8_t y = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
            lumaSum += y;
            if (y < lumaMin) lumaMin = y;
            if (y > lumaMax) lumaMax = y;
        }
    }
    float lumaMean = (bw * bh > 0) ? (float)lumaSum / (bw * bh) : 0.f;
    addLog("[LUMA_STATS] min=" + std::to_string(lumaMin) +
           " max=" + std::to_string(lumaMax) +
           " mean=" + std::to_string((int)lumaMean) +
           " contrast=" + std::to_string(lumaMax - lumaMin));

    // Pass RGBA crop directly to ZXing via rowStride (zero-copy).
    // ZXing handles its own luma conversion and rotation internally.
    ReaderOptions opts;
    opts.setFormats(BarcodeFormat::Any);
    opts.setTryHarder(true);
    opts.setTryRotate(true);
    opts.setTryInvert(true);
    opts.setTryDownscale(true);
    opts.setIsPure(false);
    opts.setReturnErrors(true);
    addLog("[ZXING_OPTS] tryHarder=true tryRotate=true tryInvert=true tryDownscale=true formats=Any");

    // Verify readers are instantiated (diagnose missing ZXING_WITH_* defines)
    {
        MultiFormatReader testReader(opts);
        // The reader count is private, but we can check by trying a small decode
        addLog("[ZXING_READERS_CHECK] MultiFormatReader created (if 0 results on clear image, ZXING_WITH_* may be missing)");
    }

    const uint8_t* cropOrigin = rgba + (by * frameW + bx) * 4;
    ImageView rgbaView(cropOrigin, bw, bh, ImageFormat::RGBA, frameW * 4, 4);

    Barcodes barcodes;
    try {
        barcodes = ReadBarcodes(rgbaView, opts);
    } catch (const std::exception& ex) {
        addLog(std::string("[ZXING_THREW] ") + ex.what());
    }

    addLog("[ZXING_RGBA] total=" + std::to_string(barcodes.size()));

    // Fallback: try on full frame if crop failed
    if (barcodes.empty()) {
        addLog("[ZXING_FULL_FRAME] trying full 640x480 frame");
        ImageView fullView(rgba, frameW, frameH, ImageFormat::RGBA);
        try {
            barcodes = ReadBarcodes(fullView, opts);
        } catch (const std::exception& ex) {
            addLog(std::string("[ZXING_FULL_THREW] ") + ex.what());
        }
        addLog("[ZXING_FULL_FRAME] total=" + std::to_string(barcodes.size()));
    }

    // Release RGBA bytes
    env->ReleaseByteArrayElements(jRGBA, raw, JNI_ABORT);

    int validCount = 0, invalidCount = 0;
    for (const auto& bc : barcodes) { if (bc.isValid()) ++validCount; else ++invalidCount; }
    addLog("[ZXING_RESULTS] total=" + std::to_string(barcodes.size()) +
           " valid=" + std::to_string(validCount) +
           " invalid=" + std::to_string(invalidCount));

    struct Result { std::string fmt, text; int cx[4], cy[4]; };
    std::vector<Result> results;

    for (int bi = 0; bi < (int)barcodes.size(); ++bi) {
        const auto& bc = barcodes[bi];
        if (!bc.isValid()) {
            addLog("[ZXING_INVALID#" + std::to_string(bi) + "] format=" +
                   ToString(bc.format()) + " error=" + ToString(bc.error()));
            continue;
        }
        addLog("[ZXING_VALID#" + std::to_string(bi) + "] format=" +
               ToString(bc.format()) + " text=" + bc.text().substr(0, 40));
        Result r;
        r.fmt  = ToString(bc.format());
        r.text = bc.text();
        const auto& pos = bc.position();
        const PointI pts[4] = {
            pos.topLeft(), pos.topRight(), pos.bottomRight(), pos.bottomLeft()
        };
        for (int i = 0; i < 4; ++i) {
            // Points are in the coordinate system used by the successful strategy.
            // For strategy A/C (no manual rotation), map crop coords to frame coords.
            r.cx[i] = pts[i].x + bx;
            r.cy[i] = pts[i].y + by;
        }
        results.push_back(r);
    }

    // Build output array; when debug, prepend the log entry
    int extraRows = dbg ? 1 : 0;
    jobjectArray outer = env->NewObjectArray((jsize)(results.size() + extraRows), stringArrayClass, nullptr);

    if (dbg) {
        std::string joined;
        for (auto& l : logLines) { joined += l; joined += "\n"; }
        jobjectArray logRow = env->NewObjectArray(2, stringClass, nullptr);
        env->SetObjectArrayElement(logRow, 0, env->NewStringUTF("__log__"));
        env->SetObjectArrayElement(logRow, 1, env->NewStringUTF(joined.c_str()));
        env->SetObjectArrayElement(outer, 0, logRow);
        env->DeleteLocalRef(logRow);
    }

    for (int i = 0; i < (int)results.size(); ++i) {
        const auto& r = results[i];
        jobjectArray inner = env->NewObjectArray(10, stringClass, nullptr);
        auto set = [&](int idx, const std::string& s) {
            env->SetObjectArrayElement(inner, idx, env->NewStringUTF(s.c_str()));
        };
        set(0, r.fmt);
        set(1, r.text);
        for (int c = 0; c < 4; ++c) {
            set(2 + c * 2,     std::to_string(r.cx[c]));
            set(2 + c * 2 + 1, std::to_string(r.cy[c]));
        }
        env->SetObjectArrayElement(outer, i + extraRows, inner);
        env->DeleteLocalRef(inner);
    }
    return outer;
}

