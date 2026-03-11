// SPDX-License-Identifier: Apache-2.0
// ZXingNanoDetPlugin — iOS VisionCamera v4 Frame Processor Plugin
//
// Pipeline per frame:
//  1. CMSampleBuffer → CVPixelBuffer (YUV NV12 / BGRA)
//  2. Convert luminance plane to grayscale + copy RGBA for NanoDet
//  3. Run NanoDet ONNX inference via ORT C API → detection bounding boxes
//  4. For each box: crop grayscale ROI → ZXing ReadBarcodes
//  5. Return [{format, text, confidence, boundingBox, cornerPoints}]

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/** Load the NanoDet ONNX model from the given file path.
 *  Call once from +load or application:didFinishLaunchingWithOptions:.
 *  @return YES on success. */
BOOL ZXingNanoDetInitSession(const char* modelFilePath);

/** Release ORT resources; safe to call multiple times. */
void ZXingNanoDetReleaseSession(void);

#ifdef __cplusplus
}
#endif
