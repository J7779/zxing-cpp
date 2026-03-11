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


// Small-crop upscale helpers
// When a barcode crop is small (few pixels per module), bilinear 2x upscale
// combined with a light unsharp-mask dramatically improves decode rate.

// Threshold: if max(cropW,cropH) < this, upscale 2x before ZXing decode.
static constexpr int UPSCALE_THRESHOLD = 200;

// Bilinear 2x upscale of an RGBA sub-image into a contiguous buffer.
static std::vector<uint8_t> bilinearUpscale2x(
    const uint8_t* src, int srcW, int srcH, int srcRowStride)
{
    const int dstW = srcW * 2;
    const int dstH = srcH * 2;
    std::vector<uint8_t> dst(dstW * dstH * 4);

    for (int dy = 0; dy < dstH; ++dy) {
        float sy = dy * 0.5f;
        int y0 = std::min((int)sy, srcH - 1);
        int y1 = std::min(y0 + 1, srcH - 1);
        float fy = sy - y0;
        for (int dx = 0; dx < dstW; ++dx) {
            float sx = dx * 0.5f;
            int x0 = std::min((int)sx, srcW - 1);
            int x1 = std::min(x0 + 1, srcW - 1);
            float fx = sx - x0;

            const uint8_t* p00 = src + y0 * srcRowStride + x0 * 4;
            const uint8_t* p10 = src + y0 * srcRowStride + x1 * 4;
            const uint8_t* p01 = src + y1 * srcRowStride + x0 * 4;
            const uint8_t* p11 = src + y1 * srcRowStride + x1 * 4;

            uint8_t* out = dst.data() + (dy * dstW + dx) * 4;
            for (int c = 0; c < 4; ++c) {
                float v = p00[c] * (1 - fx) * (1 - fy)
                        + p10[c] *      fx  * (1 - fy)
                        + p01[c] * (1 - fx) *      fy
                        + p11[c] *      fx  *      fy;
                out[c] = (uint8_t)std::min(std::max((int)(v + 0.5f), 0), 255);
            }
        }
    }
    return dst;
}

// Light unsharp-mask on a contiguous RGBA buffer (in-place).
// 3x3 box blur kernel, amount 0.5.
static void unsharpMask(uint8_t* img, int w, int h)
{
    const float amount = 0.5f;
    std::vector<uint8_t> blur(w * h * 4);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int sum[3] = {0, 0, 0};
            int count = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                int ny = y + ky;
                if (ny < 0 || ny >= h) continue;
                for (int kx = -1; kx <= 1; ++kx) {
                    int nx = x + kx;
                    if (nx < 0 || nx >= w) continue;
                    const uint8_t* p = img + (ny * w + nx) * 4;
                    sum[0] += p[0]; sum[1] += p[1]; sum[2] += p[2];
                    ++count;
                }
            }
            uint8_t* out = blur.data() + (y * w + x) * 4;
            out[0] = sum[0] / count;
            out[1] = sum[1] / count;
            out[2] = sum[2] / count;
            out[3] = img[(y * w + x) * 4 + 3];
        }
    }

    for (int i = 0; i < w * h; ++i) {
        uint8_t* p = img + i * 4;
        const uint8_t* b = blur.data() + i * 4;
        for (int c = 0; c < 3; ++c) {
            float v = p[c] + amount * (p[c] - b[c]);
            p[c] = (uint8_t)std::min(std::max((int)(v + 0.5f), 0), 255);
        }
    }
}


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
    jboolean jDebug,
    jboolean jDamagedBarcode)
{
    bool dbg = (jDebug == JNI_TRUE);
    bool damaged = (jDamagedBarcode == JNI_TRUE);
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

    // Compute luma stats from RGBA for diagnostics (debug only — expensive)
    if (dbg) {
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
    }

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

    // Aggressive options only when damaged/curved barcode mode is enabled.
    // These multiply ZXing work by ~6x and must not run on every frame.
    if (damaged) {
        opts.setTryAngledScanning(true);
        opts.setTryUpscale(true);
        opts.setRelaxedLinearTolerance(true);
        opts.setMinLineCount(1);
    }
    addLog(std::string("[ZXING_OPTS] tryHarder=true tryRotate=true tryInvert=true tryDownscale=true")
           + " damaged=" + (damaged ? "true" : "false")
           + (damaged ? " tryAngledScanning=true tryUpscale=true relaxedLinearTolerance=true minLineCount=1" : "")
           + " formats=Any");

    const uint8_t* cropOrigin = rgba + (by * frameW + bx) * 4;
    ImageView rgbaView(cropOrigin, bw, bh, ImageFormat::RGBA, frameW * 4, 4);

    Barcodes barcodes;
    try {
        barcodes = ReadBarcodes(rgbaView, opts);
    } catch (const std::exception& ex) {
        addLog(std::string("[ZXING_THREW] ") + ex.what());
    }

    addLog("[ZXING_RGBA] total=" + std::to_string(barcodes.size()));

    // ── Curved-barcode fallback: scan horizontal strips (damaged mode only) ─
    // On cylindrical surfaces (bottles), the full crop is too warped.
    // Scanning narrow horizontal strips finds a "flat enough" slice.
    if (damaged && barcodes.empty() && bh > 20) {
        const int NUM_STRIPS = 5;
        const int stripH = bh / NUM_STRIPS;
        addLog("[CURVED_STRIPS] trying " + std::to_string(NUM_STRIPS) +
               " horizontal strips of height " + std::to_string(stripH));
        for (int s = 0; s < NUM_STRIPS && barcodes.empty(); ++s) {
            int stripY = s * stripH;
            const uint8_t* stripOrigin = rgba + ((by + stripY) * frameW + bx) * 4;
            // Use full width but narrow height — captures one "ring" of the cylinder
            ImageView stripView(stripOrigin, bw, stripH, ImageFormat::RGBA, frameW * 4, 4);
            try {
                barcodes = ReadBarcodes(stripView, opts);
            } catch (const std::exception& ex) {
                addLog(std::string("[CURVED_STRIP#") + std::to_string(s) + "_THREW] " + ex.what());
            }
            if (!barcodes.empty()) {
                addLog("[CURVED_STRIP#" + std::to_string(s) + "] decoded " +
                       std::to_string(barcodes.size()) + " barcode(s)");
            }
        }
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

