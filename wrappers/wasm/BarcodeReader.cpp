/*
 * Copyright 2016 Nu-book Inc.
 * Copyright 2023 Axel Waggershauser
 */
// SPDX-License-Identifier: Apache-2.0

#include "ReadBarcode.h"
#include "BinaryBitmap.h"
#include "BitMatrix.h"
#include "GlobalHistogramBinarizer.h"
#include "HybridBinarizer.h"
#include "ThresholdBinarizer.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <deque>
#include <map>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

using namespace ZXing;

// Global setting for returning debug images (default OFF)
static bool g_returnDebugImage = false;

void setReturnDebugImage(bool enable) {
	g_returnDebugImage = enable;
}

bool getReturnDebugImage() {
	return g_returnDebugImage;
}

// Consensus algorithm: requires N of M frames to agree on the same barcode
class ConsensusBuffer {
private:
	std::deque<std::string> buffer;
	int requiredFrames = 3;
	int requiredMatches = 2;

public:
	void configure(int frames, int matches) {
		requiredFrames = frames;
		requiredMatches = matches;
		clear();
	}
	
	void clear() {
		buffer.clear();
	}
	
	int getBufferSize() const {
		return static_cast<int>(buffer.size());
	}
	
	int getRequiredFrames() const {
		return requiredFrames;
	}
	
	// Add a result to buffer and return consensus result (empty string if no consensus)
	// Input: "format:text" or empty string for no detection
	std::string addAndGetConsensus(const std::string& result) {
		buffer.push_back(result);
		
		// Keep only last N frames
		while (static_cast<int>(buffer.size()) > requiredFrames) {
			buffer.pop_front();
		}
		
		// Need enough frames to make a decision
		if (static_cast<int>(buffer.size()) < requiredFrames) {
			return "__pending__";
		}
		
		// Count occurrences of each result
		std::map<std::string, int> counts;
		for (const auto& r : buffer) {
			counts[r]++;
		}
		
		// Find result that meets threshold
		for (const auto& [key, count] : counts) {
			if (count >= requiredMatches) {
				// Found consensus - clear buffer for next scan
				if (!key.empty()) {
					clear();
				}
				return key; // Return the consensus result (or empty for "no barcode")
			}
		}
		
		return "__pending__"; // No consensus yet
	}
};

// Global consensus buffer instance
static ConsensusBuffer g_consensusBuffer;

// Global setting for live debug stream
static bool g_enableLiveDebugStream = false;

// ============================================================================
// COMPREHENSIVE IMAGE PROCESSING PARAMETERS (OpenCV-style)
// ============================================================================
struct ImageProcessingParams {
	// === BLUR SECTION ===
	int blurType = 0;           // 0=None, 1=Box, 2=Gaussian, 3=Median, 4=Bilateral
	int blurRadius = 1;         // Kernel radius (actual size = 2*radius+1)
	float gaussianSigma = 1.0f; // Gaussian sigma
	float bilateralSigmaColor = 50.0f;  // Bilateral color sigma
	float bilateralSigmaSpace = 50.0f;  // Bilateral spatial sigma
	
	// === THRESHOLD SECTION ===
	int thresholdType = 0;      // 0=None, 1=Binary, 2=BinaryInv, 3=Trunc, 4=ToZero, 5=ToZeroInv, 6=Otsu, 7=AdaptiveMean, 8=AdaptiveGaussian
	int thresholdValue = 128;   // Manual threshold value (0-255)
	int adaptiveBlockSize = 11; // Block size for adaptive (must be odd)
	int adaptiveC = 2;          // Constant subtracted from mean
	
	// === CONTRAST/BRIGHTNESS ===
	float brightness = 0.0f;    // -100 to +100
	float contrast = 1.0f;      // 0.5 to 3.0
	bool histogramEqualize = false;
	bool clahe = false;         // Contrast Limited Adaptive Histogram Equalization
	int claheClipLimit = 40;    // CLAHE clip limit (x10)
	int claheTileSize = 8;      // CLAHE tile grid size
	
	// === MORPHOLOGY SECTION ===
	int morphType = 0;          // 0=None, 1=Erode, 2=Dilate, 3=Open, 4=Close, 5=Gradient, 6=TopHat, 7=BlackHat
	int morphKernelType = 0;    // 0=Rect, 1=Cross, 2=Ellipse
	int morphSize = 1;          // Kernel size
	int morphIterations = 1;    // Number of iterations
	
	// === EDGE DETECTION ===
	int edgeType = 0;           // 0=None, 1=Sobel, 2=Scharr, 3=Laplacian, 4=Canny
	int sobelKsize = 3;         // Sobel kernel size (1, 3, 5, 7)
	int cannyLow = 50;          // Canny low threshold
	int cannyHigh = 150;        // Canny high threshold
	
	// === SHARPENING ===
	bool sharpen = false;
	float sharpenAmount = 1.0f; // 0 to 5
	
	// === NOISE REDUCTION ===
	bool denoiseNLM = false;    // Non-local means denoising
	int denoiseH = 10;          // Filter strength
	
	// === GAMMA CORRECTION ===
	float gamma = 1.0f;         // 0.1 to 3.0
	
	// === INVERT ===
	bool invert = false;
	
	// === POST-BINARIZATION ===
	bool removeSmallBlobs = false;
	int minBlobSize = 10;
	
	// === KRAKEN NLBIN (Non-Linear Binarization) ===
	// Based on kraken.binarization by Benjamin Kiessling & Thomas M. Breuel
	// Apache License 2.0
	bool nlbinEnabled = false;        // Master enable for nlbin
	float nlbinThreshold = 0.5f;      // Final threshold (0.0-1.0)
	float nlbinZoom = 0.5f;           // Zoom for background estimation (0.1-1.0)
	float nlbinEscale = 1.0f;         // Scale for text region mask estimation
	float nlbinBorder = 0.1f;         // Ignore this fraction of border (0.0-0.5)
	int nlbinPerc = 80;               // Percentile for filters (1-99)
	int nlbinRange = 20;              // Range for percentile filters
	int nlbinLow = 5;                 // Low percentile for black estimation (1-50)
	int nlbinHigh = 90;               // High percentile for white estimation (50-99)
	
	// === PROCESSING ORDER ===
	// Order: Brightness/Contrast -> Gamma -> Blur -> Sharpen -> Edge -> nlbin OR Threshold -> Morphology -> Blob removal
};

static ImageProcessingParams g_imgParams;

// Set any parameter by name
void setImageParam(std::string param, float value) {
	// Blur
	if (param == "blurType") g_imgParams.blurType = static_cast<int>(value);
	else if (param == "blurRadius") g_imgParams.blurRadius = static_cast<int>(value);
	else if (param == "gaussianSigma") g_imgParams.gaussianSigma = value;
	else if (param == "bilateralSigmaColor") g_imgParams.bilateralSigmaColor = value;
	else if (param == "bilateralSigmaSpace") g_imgParams.bilateralSigmaSpace = value;
	// Threshold
	else if (param == "thresholdType") g_imgParams.thresholdType = static_cast<int>(value);
	else if (param == "thresholdValue") g_imgParams.thresholdValue = static_cast<int>(value);
	else if (param == "adaptiveBlockSize") g_imgParams.adaptiveBlockSize = static_cast<int>(value) | 1; // ensure odd
	else if (param == "adaptiveC") g_imgParams.adaptiveC = static_cast<int>(value);
	// Contrast/Brightness
	else if (param == "brightness") g_imgParams.brightness = value;
	else if (param == "contrast") g_imgParams.contrast = value;
	else if (param == "histogramEqualize") g_imgParams.histogramEqualize = value > 0.5f;
	else if (param == "clahe") g_imgParams.clahe = value > 0.5f;
	else if (param == "claheClipLimit") g_imgParams.claheClipLimit = static_cast<int>(value);
	else if (param == "claheTileSize") g_imgParams.claheTileSize = static_cast<int>(value);
	// Morphology
	else if (param == "morphType") g_imgParams.morphType = static_cast<int>(value);
	else if (param == "morphKernelType") g_imgParams.morphKernelType = static_cast<int>(value);
	else if (param == "morphSize") g_imgParams.morphSize = static_cast<int>(value);
	else if (param == "morphIterations") g_imgParams.morphIterations = static_cast<int>(value);
	// Edge
	else if (param == "edgeType") g_imgParams.edgeType = static_cast<int>(value);
	else if (param == "sobelKsize") g_imgParams.sobelKsize = static_cast<int>(value) | 1;
	else if (param == "cannyLow") g_imgParams.cannyLow = static_cast<int>(value);
	else if (param == "cannyHigh") g_imgParams.cannyHigh = static_cast<int>(value);
	// Sharpen
	else if (param == "sharpen") g_imgParams.sharpen = value > 0.5f;
	else if (param == "sharpenAmount") g_imgParams.sharpenAmount = value;
	// Denoise
	else if (param == "denoiseNLM") g_imgParams.denoiseNLM = value > 0.5f;
	else if (param == "denoiseH") g_imgParams.denoiseH = static_cast<int>(value);
	// Gamma
	else if (param == "gamma") g_imgParams.gamma = value;
	// Invert
	else if (param == "invert") g_imgParams.invert = value > 0.5f;
	// Blob removal
	else if (param == "removeSmallBlobs") g_imgParams.removeSmallBlobs = value > 0.5f;
	else if (param == "minBlobSize") g_imgParams.minBlobSize = static_cast<int>(value);
	// Kraken nlbin
	else if (param == "nlbinEnabled") g_imgParams.nlbinEnabled = value > 0.5f;
	else if (param == "nlbinThreshold") g_imgParams.nlbinThreshold = value;
	else if (param == "nlbinZoom") g_imgParams.nlbinZoom = value;
	else if (param == "nlbinEscale") g_imgParams.nlbinEscale = value;
	else if (param == "nlbinBorder") g_imgParams.nlbinBorder = value;
	else if (param == "nlbinPerc") g_imgParams.nlbinPerc = static_cast<int>(value);
	else if (param == "nlbinRange") g_imgParams.nlbinRange = static_cast<int>(value);
	else if (param == "nlbinLow") g_imgParams.nlbinLow = static_cast<int>(value);
	else if (param == "nlbinHigh") g_imgParams.nlbinHigh = static_cast<int>(value);
}

void configureConsensus(int frames, int matches) {
	g_consensusBuffer.configure(frames, matches);
}

void clearConsensus() {
	g_consensusBuffer.clear();
}

int getConsensusBufferSize() {
	return g_consensusBuffer.getBufferSize();
}

int getConsensusRequiredFrames() {
	return g_consensusBuffer.getRequiredFrames();
}

void setLiveDebugStream(bool enable) {
	g_enableLiveDebugStream = enable;
}

bool getLiveDebugStream() {
	return g_enableLiveDebugStream;
}

// ============================================================================
// IMAGE PROCESSING ALGORITHMS
// ============================================================================

// --- BLUR ALGORITHMS ---

// Box blur (average filter)
static void applyBoxBlur(uint8_t* data, int width, int height, int radius) {
	if (radius < 1) return;
	std::vector<uint8_t> temp(width * height);
	
	// Horizontal pass
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int sum = 0, count = 0;
			for (int dx = -radius; dx <= radius; ++dx) {
				int nx = x + dx;
				if (nx >= 0 && nx < width) {
					sum += data[y * width + nx];
					count++;
				}
			}
			temp[y * width + x] = static_cast<uint8_t>(sum / count);
		}
	}
	
	// Vertical pass
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int sum = 0, count = 0;
			for (int dy = -radius; dy <= radius; ++dy) {
				int ny = y + dy;
				if (ny >= 0 && ny < height) {
					sum += temp[ny * width + x];
					count++;
				}
			}
			data[y * width + x] = static_cast<uint8_t>(sum / count);
		}
	}
}

// Gaussian blur
static void applyGaussianBlur(uint8_t* data, int width, int height, int radius, float sigma) {
	if (radius < 1) return;
	
	// Generate 1D Gaussian kernel
	int ksize = 2 * radius + 1;
	std::vector<float> kernel(ksize);
	float sum = 0;
	for (int i = 0; i < ksize; ++i) {
		float x = static_cast<float>(i - radius);
		kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
		sum += kernel[i];
	}
	for (int i = 0; i < ksize; ++i) kernel[i] /= sum;
	
	std::vector<uint8_t> temp(width * height);
	
	// Horizontal pass
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float val = 0;
			for (int i = 0; i < ksize; ++i) {
				int nx = std::max(0, std::min(width - 1, x + i - radius));
				val += data[y * width + nx] * kernel[i];
			}
			temp[y * width + x] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
		}
	}
	
	// Vertical pass
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float val = 0;
			for (int i = 0; i < ksize; ++i) {
				int ny = std::max(0, std::min(height - 1, y + i - radius));
				val += temp[ny * width + x] * kernel[i];
			}
			data[y * width + x] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
		}
	}
}

// Median blur
static void applyMedianBlur(uint8_t* data, int width, int height, int radius) {
	if (radius < 1) return;
	std::vector<uint8_t> result(width * height);
	int ksize = 2 * radius + 1;
	int numPixels = ksize * ksize;
	std::vector<uint8_t> neighbors(numPixels);
	
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = 0;
			for (int dy = -radius; dy <= radius; ++dy) {
				for (int dx = -radius; dx <= radius; ++dx) {
					int ny = std::max(0, std::min(height - 1, y + dy));
					int nx = std::max(0, std::min(width - 1, x + dx));
					neighbors[idx++] = data[ny * width + nx];
				}
			}
			std::nth_element(neighbors.begin(), neighbors.begin() + numPixels / 2, neighbors.end());
			result[y * width + x] = neighbors[numPixels / 2];
		}
	}
	std::copy(result.begin(), result.end(), data);
}

// Bilateral filter (edge-preserving blur)
static void applyBilateralFilter(uint8_t* data, int width, int height, int radius, float sigmaColor, float sigmaSpace) {
	if (radius < 1) return;
	std::vector<uint8_t> result(width * height);
	
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float sum = 0, weightSum = 0;
			uint8_t centerVal = data[y * width + x];
			
			for (int dy = -radius; dy <= radius; ++dy) {
				for (int dx = -radius; dx <= radius; ++dx) {
					int ny = std::max(0, std::min(height - 1, y + dy));
					int nx = std::max(0, std::min(width - 1, x + dx));
					uint8_t neighborVal = data[ny * width + nx];
					
					float spatialDist = static_cast<float>(dx * dx + dy * dy);
					float colorDist = static_cast<float>((centerVal - neighborVal) * (centerVal - neighborVal));
					
					float weight = std::exp(-spatialDist / (2 * sigmaSpace * sigmaSpace) 
					                       - colorDist / (2 * sigmaColor * sigmaColor));
					sum += neighborVal * weight;
					weightSum += weight;
				}
			}
			result[y * width + x] = static_cast<uint8_t>(sum / weightSum);
		}
	}
	std::copy(result.begin(), result.end(), data);
}

// --- BRIGHTNESS/CONTRAST ---

static void applyBrightnessContrast(uint8_t* data, int width, int height, float brightness, float contrast) {
	for (int i = 0; i < width * height; ++i) {
		float val = data[i];
		val = (val - 128) * contrast + 128 + brightness;
		data[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
	}
}

// --- GAMMA CORRECTION ---

static void applyGamma(uint8_t* data, int width, int height, float gamma) {
	if (gamma == 1.0f) return;
	float invGamma = 1.0f / gamma;
	for (int i = 0; i < width * height; ++i) {
		float normalized = data[i] / 255.0f;
		data[i] = static_cast<uint8_t>(std::pow(normalized, invGamma) * 255.0f);
	}
}

// --- HISTOGRAM EQUALIZATION ---

static void applyHistogramEqualization(uint8_t* data, int width, int height) {
	// Calculate histogram
	int hist[256] = {0};
	for (int i = 0; i < width * height; ++i) {
		hist[data[i]]++;
	}
	
	// Calculate CDF
	int cdf[256];
	cdf[0] = hist[0];
	for (int i = 1; i < 256; ++i) {
		cdf[i] = cdf[i-1] + hist[i];
	}
	
	// Find min CDF
	int cdfMin = 0;
	for (int i = 0; i < 256; ++i) {
		if (cdf[i] > 0) { cdfMin = cdf[i]; break; }
	}
	
	// Apply equalization
	int total = width * height;
	for (int i = 0; i < width * height; ++i) {
		data[i] = static_cast<uint8_t>(255.0f * (cdf[data[i]] - cdfMin) / (total - cdfMin));
	}
}

// --- CLAHE (Contrast Limited Adaptive Histogram Equalization) ---

static void applyCLAHE(uint8_t* data, int width, int height, int clipLimit, int tileSize) {
	if (tileSize < 2) tileSize = 2;
	
	int tilesX = (width + tileSize - 1) / tileSize;
	int tilesY = (height + tileSize - 1) / tileSize;
	
	std::vector<uint8_t> result(width * height);
	
	for (int ty = 0; ty < tilesY; ++ty) {
		for (int tx = 0; tx < tilesX; ++tx) {
			int x0 = tx * tileSize;
			int y0 = ty * tileSize;
			int x1 = std::min(x0 + tileSize, width);
			int y1 = std::min(y0 + tileSize, height);
			
			// Calculate histogram for this tile
			int hist[256] = {0};
			int tilePixels = 0;
			for (int y = y0; y < y1; ++y) {
				for (int x = x0; x < x1; ++x) {
					hist[data[y * width + x]]++;
					tilePixels++;
				}
			}
			
			// Clip histogram
			int excess = 0;
			int limit = clipLimit * tilePixels / 256;
			for (int i = 0; i < 256; ++i) {
				if (hist[i] > limit) {
					excess += hist[i] - limit;
					hist[i] = limit;
				}
			}
			
			// Redistribute excess
			int redistrib = excess / 256;
			for (int i = 0; i < 256; ++i) {
				hist[i] += redistrib;
			}
			
			// Calculate CDF
			int cdf[256];
			cdf[0] = hist[0];
			for (int i = 1; i < 256; ++i) {
				cdf[i] = cdf[i-1] + hist[i];
			}
			
			// Apply to tile
			for (int y = y0; y < y1; ++y) {
				for (int x = x0; x < x1; ++x) {
					result[y * width + x] = static_cast<uint8_t>(255.0f * cdf[data[y * width + x]] / cdf[255]);
				}
			}
		}
	}
	std::copy(result.begin(), result.end(), data);
}

// --- SHARPENING ---

static void applySharpen(uint8_t* data, int width, int height, float amount) {
	std::vector<uint8_t> blurred(width * height);
	std::copy(data, data + width * height, blurred.begin());
	applyGaussianBlur(blurred.data(), width, height, 1, 1.0f);
	
	for (int i = 0; i < width * height; ++i) {
		float diff = static_cast<float>(data[i]) - static_cast<float>(blurred[i]);
		float sharpened = data[i] + diff * amount;
		data[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, sharpened)));
	}
}

// --- THRESHOLDING (OpenCV-style) ---

// Otsu's threshold calculation
static int calculateOtsuThreshold(const uint8_t* data, int width, int height) {
	// Calculate histogram
	int hist[256] = {0};
	int total = width * height;
	for (int i = 0; i < total; ++i) {
		hist[data[i]]++;
	}
	
	float sum = 0;
	for (int i = 0; i < 256; ++i) sum += i * hist[i];
	
	float sumB = 0, wB = 0, wF = 0;
	float maxVar = 0;
	int threshold = 0;
	
	for (int t = 0; t < 256; ++t) {
		wB += hist[t];
		if (wB == 0) continue;
		wF = total - wB;
		if (wF == 0) break;
		
		sumB += t * hist[t];
		float mB = sumB / wB;
		float mF = (sum - sumB) / wF;
		
		float varBetween = wB * wF * (mB - mF) * (mB - mF);
		if (varBetween > maxVar) {
			maxVar = varBetween;
			threshold = t;
		}
	}
	return threshold;
}

// Apply thresholding
static void applyThreshold(uint8_t* data, int width, int height, int type, int thresh, int blockSize, int C) {
	if (type == 0) return; // None
	
	int total = width * height;
	
	// Otsu - calculate threshold automatically
	if (type == 6) {
		thresh = calculateOtsuThreshold(data, width, height);
		type = 1; // Use binary threshold with Otsu value
	}
	
	// Adaptive Mean
	if (type == 7) {
		std::vector<uint8_t> result(total);
		int halfBlock = blockSize / 2;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				int sum = 0, count = 0;
				for (int dy = -halfBlock; dy <= halfBlock; ++dy) {
					for (int dx = -halfBlock; dx <= halfBlock; ++dx) {
						int ny = std::max(0, std::min(height - 1, y + dy));
						int nx = std::max(0, std::min(width - 1, x + dx));
						sum += data[ny * width + nx];
						count++;
					}
				}
				int localThresh = sum / count - C;
				result[y * width + x] = (data[y * width + x] > localThresh) ? 255 : 0;
			}
		}
		std::copy(result.begin(), result.end(), data);
		return;
	}
	
	// Adaptive Gaussian
	if (type == 8) {
		std::vector<uint8_t> blurred(total);
		std::copy(data, data + total, blurred.begin());
		applyGaussianBlur(blurred.data(), width, height, blockSize / 2, static_cast<float>(blockSize) / 6.0f);
		
		for (int i = 0; i < total; ++i) {
			int localThresh = blurred[i] - C;
			data[i] = (data[i] > localThresh) ? 255 : 0;
		}
		return;
	}
	
	// Simple thresholding types
	for (int i = 0; i < total; ++i) {
		uint8_t val = data[i];
		switch (type) {
			case 1: // Binary
				data[i] = (val > thresh) ? 255 : 0;
				break;
			case 2: // Binary Inverted
				data[i] = (val > thresh) ? 0 : 255;
				break;
			case 3: // Truncate
				data[i] = (val > thresh) ? thresh : val;
				break;
			case 4: // To Zero
				data[i] = (val > thresh) ? val : 0;
				break;
			case 5: // To Zero Inverted
				data[i] = (val > thresh) ? 0 : val;
				break;
		}
	}
}

// --- MORPHOLOGICAL OPERATIONS ---

static void getMorphKernel(std::vector<bool>& kernel, int size, int type) {
	int ksize = 2 * size + 1;
	kernel.resize(ksize * ksize);
	
	for (int y = 0; y < ksize; ++y) {
		for (int x = 0; x < ksize; ++x) {
			bool inKernel = false;
			switch (type) {
				case 0: // Rect
					inKernel = true;
					break;
				case 1: // Cross
					inKernel = (x == size || y == size);
					break;
				case 2: // Ellipse
					{
						float dx = static_cast<float>(x - size) / size;
						float dy = static_cast<float>(y - size) / size;
						inKernel = (dx * dx + dy * dy <= 1.0f);
					}
					break;
			}
			kernel[y * ksize + x] = inKernel;
		}
	}
}

static void applyErode(uint8_t* data, int width, int height, const std::vector<bool>& kernel, int ksize) {
	std::vector<uint8_t> result(width * height);
	int radius = ksize / 2;
	
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			uint8_t minVal = 255;
			for (int ky = 0; ky < ksize; ++ky) {
				for (int kx = 0; kx < ksize; ++kx) {
					if (!kernel[ky * ksize + kx]) continue;
					int ny = std::max(0, std::min(height - 1, y + ky - radius));
					int nx = std::max(0, std::min(width - 1, x + kx - radius));
					minVal = std::min(minVal, data[ny * width + nx]);
				}
			}
			result[y * width + x] = minVal;
		}
	}
	std::copy(result.begin(), result.end(), data);
}

static void applyDilate(uint8_t* data, int width, int height, const std::vector<bool>& kernel, int ksize) {
	std::vector<uint8_t> result(width * height);
	int radius = ksize / 2;
	
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			uint8_t maxVal = 0;
			for (int ky = 0; ky < ksize; ++ky) {
				for (int kx = 0; kx < ksize; ++kx) {
					if (!kernel[ky * ksize + kx]) continue;
					int ny = std::max(0, std::min(height - 1, y + ky - radius));
					int nx = std::max(0, std::min(width - 1, x + kx - radius));
					maxVal = std::max(maxVal, data[ny * width + nx]);
				}
			}
			result[y * width + x] = maxVal;
		}
	}
	std::copy(result.begin(), result.end(), data);
}

static void applyMorphology(uint8_t* data, int width, int height, int type, int kernelType, int size, int iterations) {
	if (type == 0 || size < 1) return;
	
	std::vector<bool> kernel;
	int ksize = 2 * size + 1;
	getMorphKernel(kernel, size, kernelType);
	
	for (int iter = 0; iter < iterations; ++iter) {
		switch (type) {
			case 1: // Erode
				applyErode(data, width, height, kernel, ksize);
				break;
			case 2: // Dilate
				applyDilate(data, width, height, kernel, ksize);
				break;
			case 3: // Open (Erode then Dilate)
				applyErode(data, width, height, kernel, ksize);
				applyDilate(data, width, height, kernel, ksize);
				break;
			case 4: // Close (Dilate then Erode)
				applyDilate(data, width, height, kernel, ksize);
				applyErode(data, width, height, kernel, ksize);
				break;
			case 5: // Gradient (Dilate - Erode)
				{
					std::vector<uint8_t> dilated(data, data + width * height);
					std::vector<uint8_t> eroded(data, data + width * height);
					applyDilate(dilated.data(), width, height, kernel, ksize);
					applyErode(eroded.data(), width, height, kernel, ksize);
					for (int i = 0; i < width * height; ++i) {
						data[i] = static_cast<uint8_t>(std::abs(dilated[i] - eroded[i]));
					}
				}
				break;
			case 6: // Top Hat (Original - Open)
				{
					std::vector<uint8_t> opened(data, data + width * height);
					applyErode(opened.data(), width, height, kernel, ksize);
					applyDilate(opened.data(), width, height, kernel, ksize);
					for (int i = 0; i < width * height; ++i) {
						data[i] = static_cast<uint8_t>(std::max(0, data[i] - opened[i]));
					}
				}
				break;
			case 7: // Black Hat (Close - Original)
				{
					std::vector<uint8_t> original(data, data + width * height);
					applyDilate(data, width, height, kernel, ksize);
					applyErode(data, width, height, kernel, ksize);
					for (int i = 0; i < width * height; ++i) {
						data[i] = static_cast<uint8_t>(std::max(0, data[i] - original[i]));
					}
				}
				break;
		}
	}
}

// --- EDGE DETECTION ---

static void applySobel(uint8_t* data, int width, int height, int ksize) {
	std::vector<float> gx(width * height), gy(width * height);
	
	// Sobel kernels
	float sobelX3[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	float sobelY3[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
	
	int radius = ksize / 2;
	
	for (int y = radius; y < height - radius; ++y) {
		for (int x = radius; x < width - radius; ++x) {
			float sumX = 0, sumY = 0;
			for (int ky = 0; ky < ksize; ++ky) {
				for (int kx = 0; kx < ksize; ++kx) {
					int kidx = ky * ksize + kx;
					float val = data[(y + ky - radius) * width + (x + kx - radius)];
					sumX += val * sobelX3[kidx];
					sumY += val * sobelY3[kidx];
				}
			}
			gx[y * width + x] = sumX;
			gy[y * width + x] = sumY;
		}
	}
	
	// Compute magnitude
	for (int i = 0; i < width * height; ++i) {
		float mag = std::sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
		data[i] = static_cast<uint8_t>(std::min(255.0f, mag));
	}
}

static void applyLaplacian(uint8_t* data, int width, int height) {
	std::vector<uint8_t> result(width * height);
	float kernel[9] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
	
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			float sum = 0;
			for (int ky = 0; ky < 3; ++ky) {
				for (int kx = 0; kx < 3; ++kx) {
					sum += data[(y + ky - 1) * width + (x + kx - 1)] * kernel[ky * 3 + kx];
				}
			}
			result[y * width + x] = static_cast<uint8_t>(std::min(255.0f, std::abs(sum)));
		}
	}
	std::copy(result.begin(), result.end(), data);
}

static void applyCanny(uint8_t* data, int width, int height, int lowThresh, int highThresh) {
	// Simplified Canny: Gaussian blur + Sobel + Non-max suppression + Hysteresis
	applyGaussianBlur(data, width, height, 1, 1.4f);
	
	std::vector<float> gx(width * height), gy(width * height), mag(width * height);
	std::vector<float> direction(width * height);
	
	// Sobel
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			float sumX = -data[(y-1)*width+(x-1)] + data[(y-1)*width+(x+1)]
			            -2*data[y*width+(x-1)] + 2*data[y*width+(x+1)]
			            -data[(y+1)*width+(x-1)] + data[(y+1)*width+(x+1)];
			float sumY = -data[(y-1)*width+(x-1)] - 2*data[(y-1)*width+x] - data[(y-1)*width+(x+1)]
			            +data[(y+1)*width+(x-1)] + 2*data[(y+1)*width+x] + data[(y+1)*width+(x+1)];
			gx[y * width + x] = sumX;
			gy[y * width + x] = sumY;
			mag[y * width + x] = std::sqrt(sumX * sumX + sumY * sumY);
			direction[y * width + x] = std::atan2(sumY, sumX);
		}
	}
	
	// Non-maximum suppression
	std::vector<uint8_t> result(width * height, 0);
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			float angle = direction[y * width + x] * 180.0f / 3.14159f;
			if (angle < 0) angle += 180;
			
			float m = mag[y * width + x];
			float m1, m2;
			
			if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
				m1 = mag[y * width + (x - 1)];
				m2 = mag[y * width + (x + 1)];
			} else if (angle >= 22.5 && angle < 67.5) {
				m1 = mag[(y - 1) * width + (x + 1)];
				m2 = mag[(y + 1) * width + (x - 1)];
			} else if (angle >= 67.5 && angle < 112.5) {
				m1 = mag[(y - 1) * width + x];
				m2 = mag[(y + 1) * width + x];
			} else {
				m1 = mag[(y - 1) * width + (x - 1)];
				m2 = mag[(y + 1) * width + (x + 1)];
			}
			
			if (m >= m1 && m >= m2 && m > lowThresh) {
				result[y * width + x] = (m > highThresh) ? 255 : 128;
			}
		}
	}
	
	// Hysteresis
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			if (result[y * width + x] == 128) {
				bool hasStrongNeighbor = false;
				for (int dy = -1; dy <= 1; ++dy) {
					for (int dx = -1; dx <= 1; ++dx) {
						if (result[(y + dy) * width + (x + dx)] == 255) {
							hasStrongNeighbor = true;
							break;
						}
					}
					if (hasStrongNeighbor) break;
				}
				result[y * width + x] = hasStrongNeighbor ? 255 : 0;
			}
		}
	}
	
	std::copy(result.begin(), result.end(), data);
}

// --- BLOB REMOVAL ---

static void removeSmallBlobs(uint8_t* data, int width, int height, int minSize) {
	if (minSize < 1) return;
	std::vector<bool> visited(width * height, false);
	std::vector<int> component;
	
	auto floodFill = [&](int startX, int startY, uint8_t targetVal) {
		component.clear();
		std::vector<std::pair<int,int>> stack;
		stack.push_back({startX, startY});
		
		while (!stack.empty()) {
			auto [x, y] = stack.back();
			stack.pop_back();
			
			if (x < 0 || x >= width || y < 0 || y >= height) continue;
			int idx = y * width + x;
			if (visited[idx] || data[idx] != targetVal) continue;
			
			visited[idx] = true;
			component.push_back(idx);
			
			stack.push_back({x+1, y});
			stack.push_back({x-1, y});
			stack.push_back({x, y+1});
			stack.push_back({x, y-1});
		}
	};
	
	// Remove small black blobs
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = y * width + x;
			if (!visited[idx] && data[idx] == 0) {
				floodFill(x, y, 0);
				if (static_cast<int>(component.size()) < minSize) {
					for (int i : component) data[i] = 255;
				}
			}
		}
	}
	
	// Remove small white blobs
	std::fill(visited.begin(), visited.end(), false);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = y * width + x;
			if (!visited[idx] && data[idx] == 255) {
				floodFill(x, y, 255);
				if (static_cast<int>(component.size()) < minSize) {
					for (int i : component) data[i] = 0;
				}
			}
		}
	}
}

// ============================================================================
// KRAKEN NLBIN - Non-Linear Binarization
// Based on kraken.binarization by Benjamin Kiessling & Thomas M. Breuel
// Licensed under the Apache License, Version 2.0
// https://github.com/mittagessen/kraken
// ============================================================================

// Bilinear zoom/rescale helper
static void zoomImage(const float* src, int srcW, int srcH, float* dst, int dstW, int dstH) {
	float scaleX = static_cast<float>(srcW) / dstW;
	float scaleY = static_cast<float>(srcH) / dstH;
	
	for (int y = 0; y < dstH; ++y) {
		float srcY = y * scaleY;
		int y0 = static_cast<int>(srcY);
		int y1 = std::min(y0 + 1, srcH - 1);
		float fy = srcY - y0;
		
		for (int x = 0; x < dstW; ++x) {
			float srcX = x * scaleX;
			int x0 = static_cast<int>(srcX);
			int x1 = std::min(x0 + 1, srcW - 1);
			float fx = srcX - x0;
			
			float v00 = src[y0 * srcW + x0];
			float v01 = src[y0 * srcW + x1];
			float v10 = src[y1 * srcW + x0];
			float v11 = src[y1 * srcW + x1];
			
			float v0 = v00 * (1 - fx) + v01 * fx;
			float v1 = v10 * (1 - fx) + v11 * fx;
			dst[y * dstW + x] = v0 * (1 - fy) + v1 * fy;
		}
	}
}

// Percentile filter (separable, does horizontal or vertical depending on mode)
static void percentileFilter1D(const float* src, float* dst, int width, int height, 
                               int perc, int range, bool horizontal) {
	std::vector<float> window;
	window.reserve(range * 2 + 1);
	
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			window.clear();
			if (horizontal) {
				for (int dx = -range; dx <= range; ++dx) {
					int nx = std::max(0, std::min(width - 1, x + dx));
					window.push_back(src[y * width + nx]);
				}
			} else {
				for (int dy = -range; dy <= range; ++dy) {
					int ny = std::max(0, std::min(height - 1, y + dy));
					window.push_back(src[ny * width + x]);
				}
			}
			std::sort(window.begin(), window.end());
			int idx = static_cast<int>((window.size() - 1) * perc / 100);
			dst[y * width + x] = window[idx];
		}
	}
}

// Gaussian blur for floats (separable)
static void gaussianBlurFloat(float* data, int width, int height, float sigma) {
	if (sigma <= 0) return;
	int radius = static_cast<int>(3 * sigma);
	if (radius < 1) radius = 1;
	
	std::vector<float> kernel(radius * 2 + 1);
	float sum = 0;
	for (int i = -radius; i <= radius; ++i) {
		kernel[i + radius] = std::exp(-(i * i) / (2 * sigma * sigma));
		sum += kernel[i + radius];
	}
	for (auto& k : kernel) k /= sum;
	
	std::vector<float> temp(width * height);
	
	// Horizontal pass
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float v = 0;
			for (int dx = -radius; dx <= radius; ++dx) {
				int nx = std::max(0, std::min(width - 1, x + dx));
				v += data[y * width + nx] * kernel[dx + radius];
			}
			temp[y * width + x] = v;
		}
	}
	
	// Vertical pass
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float v = 0;
			for (int dy = -radius; dy <= radius; ++dy) {
				int ny = std::max(0, std::min(height - 1, y + dy));
				v += temp[ny * width + x] * kernel[dy + radius];
			}
			data[y * width + x] = v;
		}
	}
}

// Binary dilation for float mask
static void binaryDilateFloat(float* data, int width, int height, int kernelW, int kernelH) {
	std::vector<float> temp(width * height);
	std::copy(data, data + width * height, temp.begin());
	
	int kx = kernelW / 2;
	int ky = kernelH / 2;
	
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float maxVal = 0;
			for (int dy = -ky; dy <= ky; ++dy) {
				for (int dx = -kx; dx <= kx; ++dx) {
					int ny = std::max(0, std::min(height - 1, y + dy));
					int nx = std::max(0, std::min(width - 1, x + dx));
					maxVal = std::max(maxVal, temp[ny * width + nx]);
				}
			}
			data[y * width + x] = maxVal;
		}
	}
}

// Main nlbin algorithm
static void applyNlbin(uint8_t* data, int width, int height,
                       float threshold, float zoom, float escale,
                       float border, int perc, int range, int low, int high) {
	// Convert to float [0,1] - rescale from byte range
	std::vector<float> image(width * height);
	for (int i = 0; i < width * height; ++i) {
		image[i] = data[i] / 255.0f;
	}
	
	// Normalize to 0-1 range
	float minVal = *std::min_element(image.begin(), image.end());
	float maxVal = *std::max_element(image.begin(), image.end());
	if (maxVal == minVal) {
		// Empty image, skip
		for (int i = 0; i < width * height; ++i) data[i] = 255;
		return;
	}
	for (auto& v : image) {
		v = (v - minVal) / (maxVal - minVal);
	}
	
	// Create zoomed image for background estimation
	int zoomW = std::max(1, static_cast<int>(width * zoom));
	int zoomH = std::max(1, static_cast<int>(height * zoom));
	std::vector<float> zoomed(zoomW * zoomH);
	zoomImage(image.data(), width, height, zoomed.data(), zoomW, zoomH);
	
	// Percentile filter on zoomed image
	std::vector<float> temp(zoomW * zoomH);
	percentileFilter1D(zoomed.data(), temp.data(), zoomW, zoomH, perc, range, true);  // horizontal
	percentileFilter1D(temp.data(), zoomed.data(), zoomW, zoomH, perc, range, false); // vertical
	
	// Scale back to original size using affine transform
	std::vector<float> background(width * height);
	zoomImage(zoomed.data(), zoomW, zoomH, background.data(), width, height);
	
	// Flatten: image - background + 1, clipped to [0,1]
	std::vector<float> flat(width * height);
	for (int i = 0; i < width * height; ++i) {
		flat[i] = std::max(0.0f, std::min(1.0f, image[i] - background[i] + 1.0f));
	}
	
	// Estimate thresholds from interior region (excluding border)
	int d0 = height, d1 = width;
	int o0 = static_cast<int>(border * d0), o1 = static_cast<int>(border * d1);
	
	// Calculate variance map for the estimation region
	std::vector<float> est;
	est.reserve((d0 - 2*o0) * (d1 - 2*o1));
	for (int y = o0; y < d0 - o0; ++y) {
		for (int x = o1; x < d1 - o1; ++x) {
			est.push_back(flat[y * width + x]);
		}
	}
	
	if (est.empty()) {
		// Fall back to entire image if border is too large
		for (int i = 0; i < width * height; ++i) {
			est.push_back(flat[i]);
		}
	}
	
	// Create variance map for mask calculation (v = est - gaussian(est, escale*20))
	int estW = d1 - 2*o1;
	int estH = d0 - 2*o0;
	if (estW < 1) estW = width;
	if (estH < 1) estH = height;
	
	std::vector<float> varMap(estW * estH);
	std::vector<float> estCopy(estW * estH);
	
	// Copy est into estCopy in 2D format
	for (int i = 0; i < estW * estH && i < static_cast<int>(est.size()); ++i) {
		estCopy[i] = est[i];
	}
	
	// Blur the estimate
	std::vector<float> blurred = estCopy;
	gaussianBlurFloat(blurred.data(), estW, estH, escale * 20.0f);
	
	// v = (est - blurred)^2, then blur again, then sqrt
	for (int i = 0; i < estW * estH; ++i) {
		float diff = estCopy[i] - blurred[i];
		varMap[i] = diff * diff;
	}
	gaussianBlurFloat(varMap.data(), estW, estH, escale * 20.0f);
	for (auto& v : varMap) v = std::sqrt(v);
	
	// Create mask: v > 0.3 * max(v)
	float varMax = *std::max_element(varMap.begin(), varMap.end());
	float varThresh = 0.3f * varMax;
	std::vector<float> mask(estW * estH);
	for (int i = 0; i < estW * estH; ++i) {
		mask[i] = varMap[i] > varThresh ? 1.0f : 0.0f;
	}
	
	// Binary dilation of mask (structure is int(escale*50) x 1 then 1 x int(escale*50))
	int dilateSize = static_cast<int>(escale * 50);
	if (dilateSize > 0) {
		binaryDilateFloat(mask.data(), estW, estH, dilateSize, 1);
		binaryDilateFloat(mask.data(), estW, estH, 1, dilateSize);
	}
	
	// Gather masked values for percentile calculation
	std::vector<float> maskedVals;
	maskedVals.reserve(estW * estH);
	for (int i = 0; i < estW * estH && i < static_cast<int>(est.size()); ++i) {
		if (mask[i] > 0.5f) {
			maskedVals.push_back(est[i]);
		}
	}
	
	if (maskedVals.empty()) {
		maskedVals = est; // Fall back to full estimate
	}
	
	// Sort for percentile calculation
	std::sort(maskedVals.begin(), maskedVals.end());
	
	// Calculate lo and hi percentiles
	int loIdx = static_cast<int>((maskedVals.size() - 1) * low / 100);
	int hiIdx = static_cast<int>((maskedVals.size() - 1) * high / 100);
	float lo = maskedVals[std::max(0, loIdx)];
	float hi = maskedVals[std::min(static_cast<int>(maskedVals.size()) - 1, hiIdx)];
	
	if (hi <= lo) hi = lo + 0.001f; // Prevent division by zero
	
	// Apply normalization and final threshold
	for (int i = 0; i < width * height; ++i) {
		float v = (flat[i] - lo) / (hi - lo);
		v = std::max(0.0f, std::min(1.0f, v));
		data[i] = v > threshold ? 255 : 0;
	}
}

// ============================================================================
// MAIN PROCESSING PIPELINE
// ============================================================================

static void processImage(uint8_t* data, int width, int height) {
	// 1. Brightness/Contrast
	if (g_imgParams.brightness != 0 || g_imgParams.contrast != 1.0f) {
		applyBrightnessContrast(data, width, height, g_imgParams.brightness, g_imgParams.contrast);
	}
	
	// 2. Gamma
	if (g_imgParams.gamma != 1.0f) {
		applyGamma(data, width, height, g_imgParams.gamma);
	}
	
	// 3. Histogram Equalization
	if (g_imgParams.histogramEqualize) {
		applyHistogramEqualization(data, width, height);
	}
	
	// 4. CLAHE
	if (g_imgParams.clahe) {
		applyCLAHE(data, width, height, g_imgParams.claheClipLimit, g_imgParams.claheTileSize);
	}
	
	// 5. Blur
	switch (g_imgParams.blurType) {
		case 1: applyBoxBlur(data, width, height, g_imgParams.blurRadius); break;
		case 2: applyGaussianBlur(data, width, height, g_imgParams.blurRadius, g_imgParams.gaussianSigma); break;
		case 3: applyMedianBlur(data, width, height, g_imgParams.blurRadius); break;
		case 4: applyBilateralFilter(data, width, height, g_imgParams.blurRadius, 
		                             g_imgParams.bilateralSigmaColor, g_imgParams.bilateralSigmaSpace); break;
	}
	
	// 6. Sharpen
	if (g_imgParams.sharpen) {
		applySharpen(data, width, height, g_imgParams.sharpenAmount);
	}
	
	// 7. Edge Detection
	switch (g_imgParams.edgeType) {
		case 1: applySobel(data, width, height, g_imgParams.sobelKsize); break;
		case 2: applySobel(data, width, height, 3); break; // Scharr uses same
		case 3: applyLaplacian(data, width, height); break;
		case 4: applyCanny(data, width, height, g_imgParams.cannyLow, g_imgParams.cannyHigh); break;
	}
	
	// 8. Invert
	if (g_imgParams.invert) {
		for (int i = 0; i < width * height; ++i) {
			data[i] = 255 - data[i];
		}
	}
	
	// 9. Binarization: Either nlbin OR standard threshold (nlbin takes priority)
	if (g_imgParams.nlbinEnabled) {
		applyNlbin(data, width, height,
		           g_imgParams.nlbinThreshold, g_imgParams.nlbinZoom, g_imgParams.nlbinEscale,
		           g_imgParams.nlbinBorder, g_imgParams.nlbinPerc, g_imgParams.nlbinRange,
		           g_imgParams.nlbinLow, g_imgParams.nlbinHigh);
	} else {
		// Standard threshold
		applyThreshold(data, width, height, g_imgParams.thresholdType, g_imgParams.thresholdValue,
		               g_imgParams.adaptiveBlockSize, g_imgParams.adaptiveC);
	}
	
	// 10. Morphology
	applyMorphology(data, width, height, g_imgParams.morphType, g_imgParams.morphKernelType,
	                g_imgParams.morphSize, g_imgParams.morphIterations);
	
	// 11. Blob Removal
	if (g_imgParams.removeSmallBlobs) {
		removeSmallBlobs(data, width, height, g_imgParams.minBlobSize);
	}
}

// Generate binarized debug image from ImageView using the specified binarizer
static std::vector<uint8_t> getBinarizedImage(const ImageView& iv, const std::string& binarizer, bool preprocess = false) {
	std::vector<uint8_t> result;
	
	int width = iv.width();
	int height = iv.height();
	std::vector<uint8_t> gray(width * height);
	
	// Convert to grayscale
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			gray[y * width + x] = *iv.data(x, y);
		}
	}
	
	// Apply full processing pipeline if enabled
	if (preprocess) {
		processImage(gray.data(), width, height);
	}
	
	// If we already did thresholding in preprocessing, skip binarizer
	if (preprocess && g_imgParams.thresholdType > 0) {
		// Image is already binarized by processImage
		result.resize(width * height * 4);
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				int idx = (y * width + x) * 4;
				uint8_t val = gray[y * width + x];
				result[idx] = val;
				result[idx+1] = val;
				result[idx+2] = val;
				result[idx+3] = 255;
			}
		}
		return result;
	}
	
	// Create ImageView from grayscale
	ImageView grayView(gray.data(), width, height, ImageFormat::Lum);
	
	// Create the appropriate binarizer
	std::unique_ptr<BinaryBitmap> bitmap;
	if (binarizer == "GlobalHistogram") {
		bitmap = std::make_unique<GlobalHistogramBinarizer>(grayView);
	} else if (binarizer == "BoolCast") {
		bitmap = std::make_unique<ThresholdBinarizer>(grayView, static_cast<uint8_t>(g_imgParams.thresholdValue));
	} else {
		bitmap = std::make_unique<HybridBinarizer>(grayView);
	}
	
	const BitMatrix* bitMatrix = bitmap->getBitMatrix();
	if (!bitMatrix) return result;
	
	// Convert to RGBA for display
	result.resize(width * height * 4);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = (y * width + x) * 4;
			uint8_t val = bitMatrix->get(x, y) ? 0 : 255;
			result[idx] = val;
			result[idx+1] = val;
			result[idx+2] = val;
			result[idx+3] = 255;
		}
	}
	
	return result;
}

// Get live debug stream image - called every frame regardless of barcode detection
emscripten::val getLiveDebugImage(int bufferPtr, int imgWidth, int imgHeight, std::string binarizer, bool preprocess) {
	thread_local const emscripten::val Uint8Array = emscripten::val::global("Uint8Array");
	
	ImageView iv(reinterpret_cast<uint8_t*>(bufferPtr), imgWidth, imgHeight, ImageFormat::RGBA);
	auto debugData = getBinarizedImage(iv, binarizer, preprocess);
	
	if (debugData.empty()) {
		return emscripten::val::null();
	}
	
	return Uint8Array.new_(emscripten::typed_memory_view(debugData.size(), debugData.data()));
}

struct ReadResult
{
	std::string format{};
	std::string text{};
	emscripten::val bytes = emscripten::val::null();
	std::string error{};
	Position position{};
	std::string symbologyIdentifier{};
	emscripten::val debugImage = emscripten::val::null(); // Binarized image as Uint8Array (RGBA)
	int debugImageWidth = 0;
	int debugImageHeight = 0;
};

// Preprocess an image for better barcode detection
static std::vector<uint8_t> preprocessForReading(const ImageView& iv) {
	int width = iv.width();
	int height = iv.height();
	std::vector<uint8_t> gray(width * height);
	
	// Convert to grayscale
	if (iv.format() == ImageFormat::Lum) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				gray[y * width + x] = *iv.data(x, y);
			}
		}
	} else {
		// RGBA - use luminance calculation
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				const uint8_t* p = iv.data(x, y);
				gray[y * width + x] = static_cast<uint8_t>(0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]);
			}
		}
	}
	
	// Apply the full processing pipeline
	processImage(gray.data(), width, height);
	
	return gray;
}

std::vector<ReadResult> readBarcodes(ImageView iv, bool tryHarder, const std::string& format, int maxSymbols, 
                                     const std::string& binarizer = "LocalAverage", bool tryAngledScanning = false,
                                     bool tryUpscale = false, bool relaxedLinearTolerance = false, bool tryRotate = true,
                                     bool preprocess = false)
{
	thread_local const emscripten::val Uint8Array = emscripten::val::global("Uint8Array");
	
	try {
		// Optionally preprocess the image
		std::vector<uint8_t> preprocessedData;
		ImageView workingView = iv;
		
		if (preprocess) {
			preprocessedData = preprocessForReading(iv);
			workingView = ImageView(preprocessedData.data(), iv.width(), iv.height(), ImageFormat::Lum);
		}
		
		ReaderOptions opts;
		opts.setTryHarder(tryHarder);
		opts.setTryRotate(tryRotate);
		opts.setTryInvert(tryHarder);
		opts.setTryDownscale(tryHarder);
		opts.setFormats(BarcodeFormatsFromString(format));
		opts.setMaxNumberOfSymbols(maxSymbols);
		
		// Set binarizer
		Binarizer selectedBinarizer = Binarizer::LocalAverage;
		if (binarizer == "GlobalHistogram")
			selectedBinarizer = Binarizer::GlobalHistogram;
		else if (binarizer == "LocalAverage")
			selectedBinarizer = Binarizer::LocalAverage;
		else if (binarizer == "BoolCast")
			selectedBinarizer = Binarizer::BoolCast;
		
		opts.setBinarizer(selectedBinarizer);
		
		// Set additional 1D barcode enhancement options
		opts.setTryAngledScanning(tryAngledScanning);
		opts.setTryUpscale(tryUpscale);
		opts.setRelaxedLinearTolerance(relaxedLinearTolerance);

		auto barcodes = ReadBarcodes(workingView, opts);
		
		// NO automatic fallbacks - user controls everything manually now

		std::vector<ReadResult> readResults{};
		readResults.reserve(std::max(size_t(1), barcodes.size()));

		// Generate debug image if enabled
		emscripten::val debugImageVal = emscripten::val::null();
		int debugW = 0, debugH = 0;
		if (g_returnDebugImage) {
			auto debugData = getBinarizedImage(iv, binarizer, preprocess);
			if (!debugData.empty()) {
				debugW = iv.width();
				debugH = iv.height();
				debugImageVal = Uint8Array.new_(emscripten::typed_memory_view(debugData.size(), debugData.data()));
			}
		}

		if (barcodes.empty()) {
			// Return empty result but with debug image if enabled
			ReadResult empty;
			empty.bytes = emscripten::val::null();
			empty.debugImage = debugImageVal;
			empty.debugImageWidth = debugW;
			empty.debugImageHeight = debugH;
			readResults.push_back(std::move(empty));
		} else {
			for (auto&& barcode : barcodes) {
				const ByteArray& bytes = barcode.bytes();
				readResults.push_back({
					ToString(barcode.format()),
					barcode.text(),
					Uint8Array.new_(emscripten::typed_memory_view(bytes.size(), bytes.data())),
					ToString(barcode.error()),
					barcode.position(),
					barcode.symbologyIdentifier(),
					debugImageVal,
					debugW,
					debugH
				});
			}
		}

		return readResults;
	} catch (const std::exception& e) {
		return {{"", "", emscripten::val::null(), e.what(), {}, {}, emscripten::val::null(), 0, 0}};
	} catch (...) {
		return {{"", "", emscripten::val::null(), "Unknown error", {}, {}, emscripten::val::null(), 0, 0}};
	}
	return {};
}

std::vector<ReadResult> readBarcodesFromImage(int bufferPtr, int bufferLength, bool tryHarder, std::string format, int maxSymbols,
                                              std::string binarizer, bool tryAngledScanning, bool tryUpscale, bool relaxedLinearTolerance, bool tryRotate = true, bool preprocess = false)
{
	int width, height, channels;
	std::unique_ptr<stbi_uc, void (*)(void*)> buffer(
		stbi_load_from_memory(reinterpret_cast<const unsigned char*>(bufferPtr), bufferLength, &width, &height, &channels, 1),
		stbi_image_free);
	if (buffer == nullptr)
		return {{"", "", {}, "Error loading image"}};

	return readBarcodes({buffer.get(), width, height, ImageFormat::Lum}, tryHarder, format, maxSymbols, binarizer, tryAngledScanning, tryUpscale, relaxedLinearTolerance, tryRotate, preprocess);
}

ReadResult readBarcodeFromImage(int bufferPtr, int bufferLength, bool tryHarder, std::string format,
                                std::string binarizer = "LocalAverage", bool tryAngledScanning = false, 
                                bool tryUpscale = false, bool relaxedLinearTolerance = false, bool tryRotate = true, bool preprocess = false)
{
	return FirstOrDefault(readBarcodesFromImage(bufferPtr, bufferLength, tryHarder, format, 1, binarizer, tryAngledScanning, tryUpscale, relaxedLinearTolerance, tryRotate, preprocess));
}

std::vector<ReadResult> readBarcodesFromPixmap(int bufferPtr, int imgWidth, int imgHeight, bool tryHarder, std::string format, int maxSymbols,
                                               std::string binarizer, bool tryAngledScanning, bool tryUpscale, bool relaxedLinearTolerance, bool tryRotate = true, bool preprocess = false)
{
	return readBarcodes({reinterpret_cast<uint8_t*>(bufferPtr), imgWidth, imgHeight, ImageFormat::RGBA}, tryHarder, format, maxSymbols, binarizer, tryAngledScanning, tryUpscale, relaxedLinearTolerance, tryRotate, preprocess);
}

ReadResult readBarcodeFromPixmap(int bufferPtr, int imgWidth, int imgHeight, bool tryHarder, std::string format,
                                 std::string binarizer = "LocalAverage", bool tryAngledScanning = false,
                                 bool tryUpscale = false, bool relaxedLinearTolerance = false, bool tryRotate = true, bool preprocess = false)
{
	return FirstOrDefault(readBarcodesFromPixmap(bufferPtr, imgWidth, imgHeight, tryHarder, format, 1, binarizer, tryAngledScanning, tryUpscale, relaxedLinearTolerance, tryRotate, preprocess));
}

// Consensus-enabled version of readBarcodeFromPixmap
// Returns result with special handling for consensus:
// - If consensus not yet reached, returns result with format="__pending__" and text showing buffer status
// - If consensus reached, returns the agreed-upon result
// - If consensus is "no barcode", returns empty result
ReadResult readBarcodeFromPixmapWithConsensus(int bufferPtr, int imgWidth, int imgHeight, bool tryHarder, std::string format,
                                              std::string binarizer, bool tryAngledScanning, bool tryUpscale, 
                                              bool relaxedLinearTolerance, bool tryRotate, bool preprocess = false)
{
	// Get raw scan result
	ReadResult rawResult = readBarcodeFromPixmap(bufferPtr, imgWidth, imgHeight, tryHarder, format, 
	                                             binarizer, tryAngledScanning, tryUpscale, 
	                                             relaxedLinearTolerance, tryRotate, preprocess);
	
	// Create key for consensus: "format:text" or empty
	std::string key = rawResult.format.empty() ? "" : (rawResult.format + ":" + rawResult.text);
	
	// Add to consensus buffer and get result
	std::string consensus = g_consensusBuffer.addAndGetConsensus(key);
	
	if (consensus == "__pending__") {
		// Still waiting for consensus
		ReadResult pending;
		pending.format = "__pending__";
		pending.text = std::to_string(g_consensusBuffer.getBufferSize()) + "/" + 
		               std::to_string(g_consensusBuffer.getRequiredFrames());
		return pending;
	}
	
	if (consensus.empty()) {
		// Consensus: no barcode found
		return ReadResult{};
	}
	
	// Consensus reached - return the raw result (which should match the consensus)
	return rawResult;
}

EMSCRIPTEN_BINDINGS(BarcodeReader)
{
	using namespace emscripten;

	value_object<ReadResult>("ReadResult")
		.field("format", &ReadResult::format)
		.field("text", &ReadResult::text)
		.field("bytes", &ReadResult::bytes)
		.field("error", &ReadResult::error)
		.field("position", &ReadResult::position)
		.field("symbologyIdentifier", &ReadResult::symbologyIdentifier)
		.field("debugImage", &ReadResult::debugImage)
		.field("debugImageWidth", &ReadResult::debugImageWidth)
		.field("debugImageHeight", &ReadResult::debugImageHeight);

	value_object<ZXing::PointI>("Point").field("x", &ZXing::PointI::x).field("y", &ZXing::PointI::y);

	value_object<ZXing::Position>("Position")
		.field("topLeft", emscripten::index<0>())
		.field("topRight", emscripten::index<1>())
		.field("bottomRight", emscripten::index<2>())
		.field("bottomLeft", emscripten::index<3>());

	register_vector<ReadResult>("vector<ReadResult>");

	function("readBarcodeFromImage", &readBarcodeFromImage);
	function("readBarcodeFromPixmap", &readBarcodeFromPixmap);

	function("readBarcodesFromImage", &readBarcodesFromImage);
	function("readBarcodesFromPixmap", &readBarcodesFromPixmap);
	
	// Consensus algorithm functions
	function("configureConsensus", &configureConsensus);
	function("clearConsensus", &clearConsensus);
	function("getConsensusBufferSize", &getConsensusBufferSize);
	function("getConsensusRequiredFrames", &getConsensusRequiredFrames);
	function("readBarcodeFromPixmapWithConsensus", &readBarcodeFromPixmapWithConsensus);
	
	// Debug image functions
	function("setReturnDebugImage", &setReturnDebugImage);
	function("getReturnDebugImage", &getReturnDebugImage);
	
	// Live debug stream functions
	function("setLiveDebugStream", &setLiveDebugStream);
	function("getLiveDebugStream", &getLiveDebugStream);
	function("getLiveDebugImage", &getLiveDebugImage);
	
	// Image processing parameter control (comprehensive OpenCV-style)
	function("setImageParam", &setImageParam);
};
