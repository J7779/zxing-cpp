#pragma once
// SPDX-License-Identifier: Apache-2.0

// NanoDet barcode detector - Self-contained C++ implementation
// Provides preprocessing, postprocessing, and model data access for the
// NanoDet-Plus barcode detection model.
//
// Usage (WASM): The embedded model data is exposed to JavaScript. The actual
// neural network inference runs via ONNX Runtime Web on the JS side. Pre/post
// processing runs in C++ for portability and performance.
//
// Usage (Native with ORT): When ZXING_USE_ONNXRUNTIME is defined, the full
// inference pipeline runs in C++ using the ONNX Runtime C API.

#include <cstdint>
#include <vector>
#include <string>

namespace ZXing {
namespace NanoDet {

// A single barcode detection from the NanoDet model
struct Detection {
    float x1, y1, x2, y2;  // Bounding box in original image coordinates
    float score;             // Confidence score [0, 1]
    int classId;             // Class index (0 = barcode typically)
};

// Preprocessing result - contains the normalized tensor and scaling info
struct PreprocessResult {
    std::vector<float> tensor;   // CHW BGR normalized tensor [1, 3, H, W]
    float scale;                  // Scale factor used for letterbox
    int padX, padY;              // Padding offset (top-left origin)
    int newWidth, newHeight;     // Resized dimensions before padding
};

// ----- Model Data Access -----

// Get pointer to the embedded ONNX model binary data
const unsigned char* GetModelData();

// Get size of the embedded ONNX model in bytes
unsigned int GetModelSize();

// ----- Preprocessing -----

// Preprocess an RGBA image for NanoDet inference.
// Performs letterbox resize to targetSize x targetSize with top-left padding,
// BGR reorder, and mean/std normalization.
// Input: RGBA pixel data, dimensions
// Output: PreprocessResult with float32 tensor in NCHW BGR format
PreprocessResult Preprocess(const uint8_t* rgbaData, int srcWidth, int srcHeight, int targetSize = 640);

// ----- Postprocessing -----

// Parse raw GFL (Generalized Focal Loss) output from NanoDet.
// outputData: flat float array from model output tensor
// numBoxes: number of anchor boxes (e.g., 3598 for 640x640 input)
// boxSize: values per box (e.g., 34 = 2 classes + 32 DFL)
// srcWidth, srcHeight: original image dimensions
// scale, padX, padY: from PreprocessResult
// targetSize: model input size (e.g., 640)
// confidence: minimum confidence threshold
// regMax: DFL regression max (default 7, meaning 8 bins)
std::vector<Detection> PostprocessGFL(
    const float* outputData, int numBoxes, int boxSize,
    int srcWidth, int srcHeight,
    float scale, int padX, int padY, int targetSize,
    float confidence = 0.3f, int regMax = 7);

// Parse already-decoded output format [x1, y1, x2, y2, score, class_id]
std::vector<Detection> PostprocessDecoded(
    const float* outputData, int numBoxes, int boxSize,
    int srcWidth, int srcHeight,
    float scale, int padX, int padY,
    float confidence = 0.3f);

// Apply Non-Maximum Suppression to remove overlapping detections
std::vector<Detection> ApplyNMS(const std::vector<Detection>& detections, float iouThreshold = 0.45f);

// ----- Full Pipeline (when ONNX Runtime is available) -----

#ifdef ZXING_USE_ONNXRUNTIME

// Initialize the ONNX Runtime session using the embedded model.
// Call once at startup. Returns true on success.
bool InitSession();

// Release the ONNX Runtime session and resources.
void ReleaseSession();

// Check if the session is initialized and ready for inference.
bool IsSessionReady();

// Run the full detection pipeline: preprocess -> inference -> postprocess.
// Input: RGBA pixel data and dimensions.
// Returns detected barcode bounding boxes.
std::vector<Detection> Detect(
    const uint8_t* rgbaData, int width, int height,
    float confidence = 0.3f, int targetSize = 640);

#endif // ZXING_USE_ONNXRUNTIME

} // namespace NanoDet
} // namespace ZXing
