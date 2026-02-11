// SPDX-License-Identifier: Apache-2.0
// NanoDet barcode detector - C++ implementation of preprocessing and postprocessing.
// When ZXING_USE_ONNXRUNTIME is defined, also includes full inference pipeline.

#include "NanoDet.h"
#include "NanoDetModelData.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#ifdef ZXING_USE_ONNXRUNTIME
#include <onnxruntime_c_api.h>
#endif

namespace ZXing {
namespace NanoDet {

// ============================================================================
// Model Data Access
// ============================================================================

const unsigned char* GetModelData()
{
    return NANODET_MODEL_DATA;
}

unsigned int GetModelSize()
{
    return NANODET_MODEL_SIZE;
}

// ============================================================================
// Preprocessing
// ============================================================================

// NanoDet BGR normalization constants
static constexpr float MEAN_B = 103.53f;
static constexpr float MEAN_G = 116.28f;
static constexpr float MEAN_R = 123.675f;
static constexpr float STD_B  = 57.375f;
static constexpr float STD_G  = 57.12f;
static constexpr float STD_R  = 58.395f;

PreprocessResult Preprocess(const uint8_t* rgbaData, int srcWidth, int srcHeight, int targetSize)
{
    PreprocessResult result;

    // Calculate letterbox scaling (maintain aspect ratio)
    float scaleW = static_cast<float>(targetSize) / srcWidth;
    float scaleH = static_cast<float>(targetSize) / srcHeight;
    result.scale = std::min(scaleW, scaleH);
    result.newWidth = static_cast<int>(std::round(srcWidth * result.scale));
    result.newHeight = static_cast<int>(std::round(srcHeight * result.scale));

    // NanoDet uses top-left padding (not centered)
    result.padX = 0;
    result.padY = 0;

    // Allocate output tensor: 3 channels x targetSize x targetSize
    int planeSize = targetSize * targetSize;
    result.tensor.resize(3 * planeSize);

    // Fill with gray (128) normalized values for padding
    float grayB = (128.0f - MEAN_B) / STD_B;
    float grayG = (128.0f - MEAN_G) / STD_G;
    float grayR = (128.0f - MEAN_R) / STD_R;
    std::fill(result.tensor.begin(), result.tensor.begin() + planeSize, grayB);
    std::fill(result.tensor.begin() + planeSize, result.tensor.begin() + 2 * planeSize, grayG);
    std::fill(result.tensor.begin() + 2 * planeSize, result.tensor.end(), grayR);

    // Bilinear resize and normalize into tensor
    float invScaleX = static_cast<float>(srcWidth) / result.newWidth;
    float invScaleY = static_cast<float>(srcHeight) / result.newHeight;

    for (int y = 0; y < result.newHeight; ++y) {
        float srcYf = y * invScaleY;
        int y0 = static_cast<int>(srcYf);
        int y1 = std::min(y0 + 1, srcHeight - 1);
        float fy = srcYf - y0;

        for (int x = 0; x < result.newWidth; ++x) {
            float srcXf = x * invScaleX;
            int x0 = static_cast<int>(srcXf);
            int x1 = std::min(x0 + 1, srcWidth - 1);
            float fx = srcXf - x0;

            // Bilinear interpolation for each RGB channel
            auto sample = [&](int ch) -> float {
                float v00 = rgbaData[(y0 * srcWidth + x0) * 4 + ch];
                float v01 = rgbaData[(y0 * srcWidth + x1) * 4 + ch];
                float v10 = rgbaData[(y1 * srcWidth + x0) * 4 + ch];
                float v11 = rgbaData[(y1 * srcWidth + x1) * 4 + ch];
                float v0 = v00 * (1 - fx) + v01 * fx;
                float v1 = v10 * (1 - fx) + v11 * fx;
                return v0 * (1 - fy) + v1 * fy;
            };

            float r = sample(0);
            float g = sample(1);
            float b = sample(2);

            int idx = y * targetSize + x;
            // BGR channel order: B=plane0, G=plane1, R=plane2
            result.tensor[idx] = (b - MEAN_B) / STD_B;
            result.tensor[planeSize + idx] = (g - MEAN_G) / STD_G;
            result.tensor[2 * planeSize + idx] = (r - MEAN_R) / STD_R;
        }
    }

    return result;
}

// ============================================================================
// Postprocessing
// ============================================================================

static float CalculateIoU(const Detection& a, const Detection& b)
{
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);

    float intersection = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    float unionArea = areaA + areaB - intersection;

    return unionArea > 0 ? intersection / unionArea : 0.0f;
}

std::vector<Detection> ApplyNMS(const std::vector<Detection>& detections, float iouThreshold)
{
    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            if (CalculateIoU(detections[i], detections[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

// DFL (Distribution Focal Loss) decode: softmax over bins then weighted sum
static float DecodeDFL(const float* data, int regMax)
{
    // Find max for numerical stability
    float maxVal = data[0];
    for (int j = 1; j <= regMax; ++j)
        maxVal = std::max(maxVal, data[j]);

    // Softmax
    float sum = 0;
    std::vector<float> exps(regMax + 1);
    for (int j = 0; j <= regMax; ++j) {
        exps[j] = std::exp(data[j] - maxVal);
        sum += exps[j];
    }

    // Weighted sum: sum(softmax(x) * index)
    float dist = 0;
    for (int j = 0; j <= regMax; ++j)
        dist += (exps[j] / sum) * j;

    return dist;
}

// Build anchor grid for NanoDet-Plus strides
struct AnchorPoint {
    float x, y;
    int stride;
};

static std::vector<AnchorPoint> BuildAnchorGrid(int targetSize, int numBoxes)
{
    // Try different stride combinations to find the best match
    struct StrideConfig {
        std::vector<int> strides;
    };
    std::vector<StrideConfig> configs = {
        {{8, 16, 32, 64}},   // NanoDet-Plus
        {{8, 16, 32}},        // NanoDet standard
        {{8, 16, 32, 64, 128}} // Larger
    };

    std::vector<int> bestStrides = {8, 16, 32};
    int bestDiff = INT32_MAX;

    for (const auto& cfg : configs) {
        int count = 0;
        for (int s : cfg.strides) {
            int grid = (targetSize + s - 1) / s;
            count += grid * grid;
        }
        int diff = std::abs(count - numBoxes);
        if (diff < bestDiff) {
            bestDiff = diff;
            bestStrides = cfg.strides;
        }
    }

    std::vector<AnchorPoint> anchors;
    for (int stride : bestStrides) {
        int gridSize = (targetSize + stride - 1) / stride;
        for (int y = 0; y < gridSize; ++y) {
            for (int x = 0; x < gridSize; ++x) {
                anchors.push_back({
                    x * stride + stride / 2.0f,
                    y * stride + stride / 2.0f,
                    stride
                });
            }
        }
    }

    return anchors;
}

std::vector<Detection> PostprocessGFL(
    const float* outputData, int numBoxes, int boxSize,
    int srcWidth, int srcHeight,
    float scale, int padX, int padY, int targetSize,
    float confidence, int regMax)
{
    int numClasses = boxSize - 32; // e.g., 34 - 32 = 2 classes
    int regOffset = numClasses;     // DFL data starts after class scores

    // Build anchor grid
    auto anchors = BuildAnchorGrid(targetSize, numBoxes);

    std::vector<Detection> detections;

    for (int i = 0; i < numBoxes; ++i) {
        int offset = i * boxSize;

        // Class scores at the beginning (already sigmoid-ed)
        float maxScore = -1e9f;
        int maxClassId = 0;
        for (int c = 0; c < numClasses; ++c) {
            float s = outputData[offset + c];
            if (s > maxScore) {
                maxScore = s;
                maxClassId = c;
            }
        }

        if (maxScore < confidence) continue;

        // Get anchor
        float anchorX = 0, anchorY = 0;
        int stride = 8;
        if (i < static_cast<int>(anchors.size())) {
            anchorX = anchors[i].x;
            anchorY = anchors[i].y;
            stride = anchors[i].stride;
        }

        // Decode DFL regression: 4 edges × (regMax+1) bins
        int regStart = offset + regOffset;
        float distLeft   = DecodeDFL(outputData + regStart, regMax);
        float distTop    = DecodeDFL(outputData + regStart + (regMax + 1), regMax);
        float distRight  = DecodeDFL(outputData + regStart + 2 * (regMax + 1), regMax);
        float distBottom = DecodeDFL(outputData + regStart + 3 * (regMax + 1), regMax);

        // Convert distances to bbox in model coordinates
        float x1m = anchorX - distLeft * stride;
        float y1m = anchorY - distTop * stride;
        float x2m = anchorX + distRight * stride;
        float y2m = anchorY + distBottom * stride;

        // Convert from model coords to original image coords
        float x1 = (x1m - padX) / scale;
        float y1 = (y1m - padY) / scale;
        float x2 = (x2m - padX) / scale;
        float y2 = (y2m - padY) / scale;

        // Clamp
        x1 = std::max(0.0f, std::min(static_cast<float>(srcWidth), x1));
        y1 = std::max(0.0f, std::min(static_cast<float>(srcHeight), y1));
        x2 = std::max(0.0f, std::min(static_cast<float>(srcWidth), x2));
        y2 = std::max(0.0f, std::min(static_cast<float>(srcHeight), y2));

        // Skip tiny boxes
        if (x2 - x1 < 10 || y2 - y1 < 10) continue;

        detections.push_back({x1, y1, x2, y2, maxScore, maxClassId});
    }

    // Sort by score descending
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    // Limit and apply NMS
    if (detections.size() > 50)
        detections.resize(50);

    return ApplyNMS(detections, 0.45f);
}

std::vector<Detection> PostprocessDecoded(
    const float* outputData, int numBoxes, int boxSize,
    int srcWidth, int srcHeight,
    float scale, int padX, int padY,
    float confidence)
{
    std::vector<Detection> detections;

    for (int i = 0; i < numBoxes; ++i) {
        int offset = i * boxSize;

        float x1m = outputData[offset];
        float y1m = outputData[offset + 1];
        float x2m = outputData[offset + 2];
        float y2m = outputData[offset + 3];
        float score = outputData[offset + 4];
        int classId = boxSize >= 6 ? static_cast<int>(outputData[offset + 5]) : 0;

        if (score < confidence) continue;

        float x1 = (x1m - padX) / scale;
        float y1 = (y1m - padY) / scale;
        float x2 = (x2m - padX) / scale;
        float y2 = (y2m - padY) / scale;

        x1 = std::max(0.0f, std::min(static_cast<float>(srcWidth), x1));
        y1 = std::max(0.0f, std::min(static_cast<float>(srcHeight), y1));
        x2 = std::max(0.0f, std::min(static_cast<float>(srcWidth), x2));
        y2 = std::max(0.0f, std::min(static_cast<float>(srcHeight), y2));

        if (x2 - x1 < 10 || y2 - y1 < 10) continue;

        detections.push_back({x1, y1, x2, y2, score, classId});
    }

    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    if (detections.size() > 50)
        detections.resize(50);

    return ApplyNMS(detections, 0.45f);
}

// ============================================================================
// Full Pipeline (with ONNX Runtime C API)
// ============================================================================

#ifdef ZXING_USE_ONNXRUNTIME

static const OrtApi* g_ort = nullptr;
static OrtEnv* g_env = nullptr;
static OrtSession* g_session = nullptr;
static OrtSessionOptions* g_sessionOptions = nullptr;
static OrtMemoryInfo* g_memoryInfo = nullptr;

static void CheckOrtStatus(OrtStatus* status)
{
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        std::string err(msg);
        g_ort->ReleaseStatus(status);
        throw std::runtime_error("ONNX Runtime error: " + err);
    }
}

bool InitSession()
{
    if (g_session) return true; // Already initialized

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) return false;

    CheckOrtStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "NanoDet", &g_env));
    CheckOrtStatus(g_ort->CreateSessionOptions(&g_sessionOptions));
    CheckOrtStatus(g_ort->SetIntraOpNumThreads(g_sessionOptions, 1));
    CheckOrtStatus(g_ort->SetSessionGraphOptimizationLevel(g_sessionOptions, ORT_ENABLE_ALL));

    // Create session from embedded model data (in-memory)
    CheckOrtStatus(g_ort->CreateSessionFromArray(
        g_env, NANODET_MODEL_DATA, NANODET_MODEL_SIZE,
        g_sessionOptions, &g_session));

    CheckOrtStatus(g_ort->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &g_memoryInfo));

    return true;
}

void ReleaseSession()
{
    if (g_memoryInfo) { g_ort->ReleaseMemoryInfo(g_memoryInfo); g_memoryInfo = nullptr; }
    if (g_session) { g_ort->ReleaseSession(g_session); g_session = nullptr; }
    if (g_sessionOptions) { g_ort->ReleaseSessionOptions(g_sessionOptions); g_sessionOptions = nullptr; }
    if (g_env) { g_ort->ReleaseEnv(g_env); g_env = nullptr; }
}

bool IsSessionReady()
{
    return g_session != nullptr;
}

std::vector<Detection> Detect(
    const uint8_t* rgbaData, int width, int height,
    float confidence, int targetSize)
{
    if (!g_session) return {};

    // Preprocess
    auto prep = Preprocess(rgbaData, width, height, targetSize);

    // Create input tensor
    int64_t inputShape[] = {1, 3, targetSize, targetSize};
    OrtValue* inputTensor = nullptr;
    CheckOrtStatus(g_ort->CreateTensorWithDataAsOrtValue(
        g_memoryInfo,
        prep.tensor.data(), prep.tensor.size() * sizeof(float),
        inputShape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &inputTensor));

    // Run inference
    const char* inputNames[] = {"data"};  // NanoDet input name
    const char* outputNames[] = {"output"}; // NanoDet output name

    OrtValue* outputTensor = nullptr;
    CheckOrtStatus(g_ort->Run(
        g_session, nullptr,
        inputNames, (const OrtValue* const*)&inputTensor, 1,
        outputNames, 1, &outputTensor));

    // Get output tensor info
    OrtTensorTypeAndShapeInfo* outputInfo = nullptr;
    CheckOrtStatus(g_ort->GetTensorTypeAndShape(outputTensor, &outputInfo));

    size_t numDims;
    CheckOrtStatus(g_ort->GetDimensionsCount(outputInfo, &numDims));

    std::vector<int64_t> outputShape(numDims);
    CheckOrtStatus(g_ort->GetDimensions(outputInfo, outputShape.data(), numDims));
    g_ort->ReleaseTensorTypeAndShapeInfo(outputInfo);

    float* outputData = nullptr;
    CheckOrtStatus(g_ort->GetTensorMutableData(outputTensor, (void**)&outputData));

    int numBoxes = (numDims >= 2) ? static_cast<int>(outputShape[numDims >= 3 ? 1 : 0]) : 0;
    int boxSize  = (numDims >= 2) ? static_cast<int>(outputShape[numDims >= 3 ? 2 : 1]) : 0;

    // Detect output format
    std::vector<Detection> detections;
    if (boxSize == 5 || boxSize == 6) {
        detections = PostprocessDecoded(outputData, numBoxes, boxSize,
                                        width, height, prep.scale, prep.padX, prep.padY, confidence);
    } else {
        detections = PostprocessGFL(outputData, numBoxes, boxSize,
                                    width, height, prep.scale, prep.padX, prep.padY,
                                    targetSize, confidence);
    }

    g_ort->ReleaseValue(outputTensor);
    g_ort->ReleaseValue(inputTensor);

    return detections;
}

#endif // ZXING_USE_ONNXRUNTIME

} // namespace NanoDet
} // namespace ZXing
