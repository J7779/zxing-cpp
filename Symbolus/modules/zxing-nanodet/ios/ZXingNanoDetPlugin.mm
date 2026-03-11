// SPDX-License-Identifier: Apache-2.0
// ZXingNanoDetPlugin.mm
//
// VisionCamera v4 Frame Processor Plugin for ZXing + NanoDet barcode detection.
//
// Build requirements (add to podspec):
//   pod 'onnxruntime-c'           (ORT C API)
//   Headers from zxing-cpp/core/src  (ReadBarcode, ImageView, …)
//   Headers from zxing-cpp/core/src/onnx (NanoDet, NanoDetModelData)
//
// The nanodet_barcode.onnx model must be added to the Xcode target as a
// bundle resource so [[NSBundle mainBundle] pathForResource:…] can find it.

#import "ZXingNanoDetPlugin.h"
#import <VisionCamera/Frame.h>
#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>

// ── ZXing C++ ────────────────────────────────────────────────────────────────
#include "ReadBarcode.h"
#include "BarcodeFormat.h"
#include "ImageView.h"

// ── NanoDet ORT pipeline ─────────────────────────────────────────────────────
#define ZXING_USE_ONNXRUNTIME 1
#include "NanoDet.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <mutex>

using namespace ZXing;

// ─────────────────────────────────────────────────────────────────────────────
// MARK:  Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a NV12 / YpCbCr8BiPlanar CVPixelBuffer to a packed RGBA byte array.
/// Returns the luma plane as the output to avoid a full RGBA copy when only
/// luminance is needed by ZXing.  The RGBA buffer is still produced for NanoDet.
static bool PixelBufferToLumaAndRGBA(
    CVPixelBufferRef pixelBuffer,
    std::vector<uint8_t>& outLuma,
    std::vector<uint8_t>& outRGBA,
    int& outWidth,
    int& outHeight)
{
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    OSType fmt = CVPixelBufferGetPixelFormatType(pixelBuffer);
    int w = (int)CVPixelBufferGetWidth(pixelBuffer);
    int h = (int)CVPixelBufferGetHeight(pixelBuffer);
    outWidth  = w;
    outHeight = h;

    if (fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
        fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
    {
        // Luma plane (Y) — single-channel, stride may differ from width
        size_t lumaStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
        const uint8_t* lumaBase = (const uint8_t*)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);

        outLuma.resize(w * h);
        for (int row = 0; row < h; ++row)
            memcpy(outLuma.data() + row * w, lumaBase + row * lumaStride, w);

        // Build RGBA from Y (grey) for NanoDet — (R=G=B=Y, A=255)
        outRGBA.resize(w * h * 4);
        for (int i = 0; i < w * h; ++i) {
            uint8_t y = outLuma[i];
            outRGBA[i * 4 + 0] = y;   // R
            outRGBA[i * 4 + 1] = y;   // G
            outRGBA[i * 4 + 2] = y;   // B
            outRGBA[i * 4 + 3] = 255; // A
        }

        CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
        return true;
    }

    if (fmt == kCVPixelFormatType_32BGRA)
    {
        size_t stride = CVPixelBufferGetBytesPerRow(pixelBuffer);
        const uint8_t* src = (const uint8_t*)CVPixelBufferGetBaseAddress(pixelBuffer);

        outRGBA.resize(w * h * 4);
        outLuma.resize(w * h);
        for (int row = 0; row < h; ++row) {
            const uint8_t* rowSrc = src + row * stride;
            uint8_t* rowRGBA = outRGBA.data() + row * w * 4;
            uint8_t* rowLuma = outLuma.data() + row * w;
            for (int x = 0; x < w; ++x) {
                uint8_t b = rowSrc[x * 4 + 0];
                uint8_t g = rowSrc[x * 4 + 1];
                uint8_t r = rowSrc[x * 4 + 2];
                rowRGBA[x * 4 + 0] = r;
                rowRGBA[x * 4 + 1] = g;
                rowRGBA[x * 4 + 2] = b;
                rowRGBA[x * 4 + 3] = 255;
                // Rec. 601 luma
                rowLuma[x] = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
            }
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
        return true;
    }

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    return false;
}

/// Rotate a luma buffer 90° clockwise.
/// Input w×h → Output h×w (outW=srcH, outH=srcW)
static std::vector<uint8_t> RotateLuma90CW(
    const std::vector<uint8_t>& src, int srcW, int srcH,
    int& outW, int& outH)
{
    outW = srcH;
    outH = srcW;
    std::vector<uint8_t> dst((size_t)(srcW * srcH));
    for (int new_row = 0; new_row < outH; ++new_row)
        for (int new_col = 0; new_col < outW; ++new_col)
            dst[new_row * outW + new_col] = src[(srcH - 1 - new_col) * srcW + new_row];
    return dst;
}

/// Extract a rectangular sub-region from the luma buffer (clipped to frame bounds).
static std::vector<uint8_t> CropLuma(
    const uint8_t* luma, int frameW, int frameH,
    int x, int y, int w, int h)
{
    int x0 = std::max(0, x);
    int y0 = std::max(0, y);
    int x1 = std::min(frameW, x + w);
    int y1 = std::min(frameH, y + h);
    int cw = x1 - x0;
    int ch = y1 - y0;
    if (cw <= 0 || ch <= 0) return {};

    std::vector<uint8_t> crop(cw * ch);
    for (int row = 0; row < ch; ++row)
        memcpy(crop.data() + row * cw, luma + (y0 + row) * frameW + x0, cw);
    return crop;
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK:  Plugin implementation
// ─────────────────────────────────────────────────────────────────────────────

@implementation ZXingNanoDetPlugin
{
    BOOL _sessionReady;
    std::mutex _mutex;
}

+ (void)load
{
    [FrameProcessorPluginRegistry addFrameProcessorPlugin:@"detectBarcodes"
                                        withInitializer:^FrameProcessorPlugin*(
                                            VisionCameraProxy* proxy,
                                            NSDictionary<NSString*, id>* options)
    {
        return [[ZXingNanoDetPlugin alloc] initWithProxy:proxy withOptions:options];
    }];
}

- (instancetype)initWithProxy:(VisionCameraProxy*)proxy
                  withOptions:(NSDictionary<NSString*, id>*)options
{
    self = [super initWithProxy:proxy withOptions:options];
    if (self) {
        _sessionReady = NO;
        [self _initORT];
    }
    return self;
}

- (void)_initORT
{
    NSString* modelPath = [[NSBundle mainBundle] pathForResource:@"nanodet_barcode"
                                                          ofType:@"onnx"];
    if (!modelPath) {
        NSLog(@"[ZXingNanoDetPlugin] nanodet_barcode.onnx not found in main bundle. "
              @"Add it as a bundle resource in Xcode or via expo-asset.");
        return;
    }

    std::lock_guard<std::mutex> lock(_mutex);
    if (NanoDet::InitSession()) {
        _sessionReady = YES;
        NSLog(@"[ZXingNanoDetPlugin] NanoDet ORT session initialized.");
    } else {
        NSLog(@"[ZXingNanoDetPlugin] Failed to initialize NanoDet ORT session.");
    }
}

- (id _Nullable)callback:(Frame*)frame
           withArguments:(NSDictionary<NSString*, id>* _Nullable)args
{
    std::lock_guard<std::mutex> lock(_mutex);

    // ── Parse JS options ────────────────────────────────────────────────────
    float confidence  = 0.35f;
    int   modelSize   = 416;
    int   maxDet      = 10;
    BOOL  debugMode   = NO;

    if (args[@"confidence"])     confidence = [args[@"confidence"] floatValue];
    if (args[@"modelInputSize"]) modelSize  = [args[@"modelInputSize"] intValue];
    if (args[@"maxDetections"])  maxDet     = [args[@"maxDetections"] intValue];
    if (args[@"debug"])          debugMode  = [args[@"debug"] boolValue];

    // Accumulate per-frame log lines
    NSMutableArray<NSString*>* frameLog = [NSMutableArray array];
    auto addLog = [&](NSString* line) {
        NSLog(@"[ZXingNanoDet] %@", line);
        if (debugMode) [frameLog addObject:line];
    };

    // ── Extract pixel buffer ─────────────────────────────────────────────────
    CMSampleBufferRef sampleBuffer = frame.buffer;
    CVPixelBufferRef  pixelBuffer  = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (!pixelBuffer) {
        addLog(@"[ERROR] no pixel buffer in frame");
        return debugMode ? @[@{@"format":@"__debug__",@"text":@"",@"confidence":@0,
            @"boundingBox":@{@"x":@0,@"y":@0,@"width":@0,@"height":@0},
            @"cornerPoints":@[],@"debugLogs":frameLog}] : @[];
    }

    OSType pixFmt = CVPixelBufferGetPixelFormatType(pixelBuffer);
    std::vector<uint8_t> luma, rgba;
    int frameW = 0, frameH = 0;
    if (!PixelBufferToLumaAndRGBA(pixelBuffer, luma, rgba, frameW, frameH)) {
        addLog([NSString stringWithFormat:@"[ERROR] unsupported pixel format: %u", (unsigned)pixFmt]);
        return debugMode ? @[@{@"format":@"__debug__",@"text":@"",@"confidence":@0,
            @"boundingBox":@{@"x":@0,@"y":@0,@"width":@0,@"height":@0},
            @"cornerPoints":@[],@"debugLogs":frameLog}] : @[];
    }

    addLog([NSString stringWithFormat:@"[FRAME] %dx%d landscape=%@ pixFmt=%u sessionReady=%@",
            frameW, frameH, frameW > frameH ? @"YES" : @"NO",
            (unsigned)pixFmt, _sessionReady ? @"YES" : @"NO"]);

    // ── NanoDet inference ────────────────────────────────────────────────────
    std::vector<NanoDet::Detection> nanoDetections;

    if (_sessionReady) {
        nanoDetections = NanoDet::Detect(rgba.data(), frameW, frameH,
                                         confidence, modelSize);
        addLog([NSString stringWithFormat:@"[NANODET] %zu detection(s) at confidence>=%.2f modelSize=%d",
                nanoDetections.size(), confidence, modelSize]);
    } else {
        NanoDet::Detection full;
        full.x1 = 0; full.y1 = 0;
        full.x2 = (float)frameW; full.y2 = (float)frameH;
        full.score = 1.0f; full.classId = 0;
        nanoDetections.push_back(full);
        addLog(@"[NANODET] session NOT ready — falling back to full-frame ZXing scan");
    }

    if ((int)nanoDetections.size() > maxDet)
        nanoDetections.resize(maxDet);

    for (int di = 0; di < (int)nanoDetections.size(); ++di) {
        const auto& d = nanoDetections[di];
        addLog([NSString stringWithFormat:
            @"[DET#%d] x1=%.0f y1=%.0f x2=%.0f y2=%.0f score=%.3f",
            di, d.x1, d.y1, d.x2, d.y2, d.score]);
    }

    // ── ZXing decode on each detection crop ─────────────────────────────────
    NSMutableArray<NSDictionary*>* results = [NSMutableArray array];

    ReaderOptions zxOpts;
    zxOpts.setFormats(BarcodeFormat::Any);
    zxOpts.setTryHarder(true);
    zxOpts.setTryRotate(true);
    zxOpts.setTryInvert(true);
    zxOpts.setReturnErrors(true);
    addLog(@"[ZXING_OPTS] tryHarder=YES tryRotate=YES tryInvert=YES formats=Any");

    for (int di = 0; di < (int)nanoDetections.size(); ++di) {
        const auto& det = nanoDetections[di];
        int bx = (int)det.x1;
        int by = (int)det.y1;
        int bw = (int)(det.x2 - det.x1);
        int bh = (int)(det.y2 - det.y1);

        // Expand 10% to avoid tight clipping
        int padX = bw / 10;
        int padY = bh / 10;
        bx = std::max(0, bx - padX);   by = std::max(0, by - padY);
        bw = std::min(frameW - bx, bw + 2 * padX);
        bh = std::min(frameH - by, bh + 2 * padY);
        addLog([NSString stringWithFormat:
            @"[CROP#%d] padded origin=(%d,%d) size=%dx%d", di, bx, by, bw, bh]);

        std::vector<uint8_t> crop = CropLuma(luma.data(), frameW, frameH,
                                              bx, by, bw, bh);
        if (crop.empty()) {
            addLog([NSString stringWithFormat:@"[CROP#%d] EMPTY after CropLuma — skipping", di]);
            continue;
        }

        // Compute luma stats on the raw crop
        if (debugMode) {
            uint8_t lMin=255, lMax=0; long lSum=0;
            for (uint8_t v : crop) { lSum+=v; if(v<lMin)lMin=v; if(v>lMax)lMax=v; }
            float lMean = crop.empty() ? 0 : (float)lSum / crop.size();
            addLog([NSString stringWithFormat:
                @"[LUMA#%d] min=%d max=%d mean=%.1f contrast=%d",
                di, lMin, lMax, lMean, lMax - lMin]);
        }

        // ── Debug: encode the ROTATED crop that ZXing actually receives ────────
        // Shown after we apply 90° CW so the debug thumbnail matches ZXing input.
        NSString* debugCropBase64 = nil;

        // Rotate 90° CW for ZXing when sensor is landscape
        int zxW = bw, zxH = bh;
        const int origBH = bh;
        std::vector<uint8_t> zxCrop = crop;
        bool cropRotated = false;
        if (frameW > frameH) {
            zxCrop = RotateLuma90CW(crop, bw, bh, zxW, zxH);
            cropRotated = true;
        }
        addLog([NSString stringWithFormat:
            @"[ROTATION#%d] applied=%@ zxing_input=%dx%d",
            di, cropRotated ? @"YES" : @"NO", zxW, zxH]);

        if (debugMode) {
            NSData* zxData = [NSData dataWithBytes:zxCrop.data() length:(NSUInteger)(zxW * zxH)];
            CGColorSpaceRef graySpace = CGColorSpaceCreateDeviceGray();
            CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)zxData);
            CGImageRef cgImage = CGImageCreate(
                (size_t)zxW, (size_t)zxH, 8, 8, (size_t)zxW,
                graySpace, kCGImageAlphaNone, provider,
                NULL, false, kCGRenderingIntentDefault);
            if (cgImage) {
                UIImage* uiImage = [UIImage imageWithCGImage:cgImage];
                NSData* jpegData = UIImageJPEGRepresentation(uiImage, 0.8f);
                if (jpegData) debugCropBase64 = [jpegData base64EncodedStringWithOptions:0];
                CGImageRelease(cgImage);
            }
            CGDataProviderRelease(provider);
            CGColorSpaceRelease(graySpace);
        }

        ImageView imageView(zxCrop.data(), zxW, zxH, ImageFormat::Lum);
        Barcodes barcodes = ReadBarcodes(imageView, zxOpts);

        int validCnt = 0, invalidCnt = 0;
        for (const auto& bc : barcodes) { if (bc.isValid()) ++validCnt; else ++invalidCnt; }
        addLog([NSString stringWithFormat:
            @"[ZXING#%d] returned %zu candidate(s): valid=%d invalid=%d",
            di, barcodes.size(), validCnt, invalidCnt]);

        for (int bi = 0; bi < (int)barcodes.size(); ++bi) {
            const auto& barcode = barcodes[bi];
            if (!barcode.isValid()) {
                addLog([NSString stringWithFormat:
                    @"[ZXING_INVALID#%d.%d] format=%s error=%s",
                    di, bi,
                    ToString(barcode.format()).c_str(),
                    ToString(barcode.error()).c_str()]);
                continue;
            }

            NSString* text   = [NSString stringWithUTF8String:barcode.text().c_str()];
            NSString* format = [NSString stringWithUTF8String:ToString(barcode.format()).c_str()];
            addLog([NSString stringWithFormat:
                @"[ZXING_VALID#%d.%d] format=%@ text=%@",
                di, bi, format, [text substringToIndex:MIN(40u,(unsigned)text.length)]]);

            NSMutableArray* corners = [NSMutableArray array];
            const auto& pos = barcode.position();
            auto addPt = [&](const PointI& p) {
                int fx = cropRotated ? (p.y + bx) : (p.x + bx);
                int fy = cropRotated ? ((origBH - 1 - p.x) + by) : (p.y + by);
                [corners addObject:@{ @"x": @(fx), @"y": @(fy) }];
            };
            addPt(pos.topLeft());
            addPt(pos.topRight());
            addPt(pos.bottomRight());
            addPt(pos.bottomLeft());

            NSMutableDictionary* resultDict = [@{
                @"format":      format,
                @"text":        text,
                @"confidence":  @(det.score),
                @"boundingBox": @{
                    @"x":      @(bx),
                    @"y":      @(by),
                    @"width":  @(bw),
                    @"height": @(bh),
                },
                @"cornerPoints": corners,
            } mutableCopy];
            if (debugMode) {
                if (debugCropBase64) resultDict[@"debugCropBase64"] = debugCropBase64;
                resultDict[@"debugLogs"] = [frameLog copy];
            }
            [results addObject:resultDict];
        }

        // If ZXing couldn't decode, still surface an UNKNOWN entry with debug info
        if (validCnt == 0) {
            NSMutableDictionary* unknownDict = [@{
                @"format":     @"UNKNOWN",
                @"text":       @"",
                @"confidence": @(det.score),
                @"boundingBox": @{
                    @"x":      @(bx),
                    @"y":      @(by),
                    @"width":  @(bw),
                    @"height": @(bh),
                },
                @"cornerPoints": @[],
            } mutableCopy];
            if (debugMode) {
                if (debugCropBase64) unknownDict[@"debugCropBase64"] = debugCropBase64;
                unknownDict[@"debugLogs"] = [frameLog copy];
            }
            [results addObject:unknownDict];
        }
    }

    // If no detections at all in debug mode, return a sentinel so logs reach JS
    if (debugMode && results.count == 0) {
        [results addObject:@{
            @"format":     @"__debug__",
            @"text":       @"",
            @"confidence": @0,
            @"boundingBox": @{@"x":@0,@"y":@0,@"width":@0,@"height":@0},
            @"cornerPoints": @[],
            @"debugLogs":  [frameLog copy],
        }];
    }

    return results;
}

@end
