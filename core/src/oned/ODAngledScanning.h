/*
* Copyright 2025 ZXing contributors
*/
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ImageView.h"
#include "Pattern.h"
#include "Quadrilateral.h"

#include <cmath>
#include <vector>

namespace ZXing::OneD {

/**
 * Helper class for scanning 1D barcodes at arbitrary angles with bilinear interpolation.
 * This enables detection of barcodes that are not perfectly horizontal or vertical.
 */
class AngledScanner
{
public:
	/**
	 * Get a pixel value using bilinear interpolation for sub-pixel accuracy.
	 * This helps when bars are only 1-2 pixels wide at distance.
	 */
	static uint8_t BilinearInterpolate(const ImageView& iv, float x, float y)
	{
		int x0 = static_cast<int>(x);
		int y0 = static_cast<int>(y);
		int x1 = std::min(x0 + 1, iv.width() - 1);
		int y1 = std::min(y0 + 1, iv.height() - 1);
		
		float fx = x - x0;
		float fy = y - y0;
		
		// Clamp to bounds
		x0 = std::max(0, std::min(x0, iv.width() - 1));
		y0 = std::max(0, std::min(y0, iv.height() - 1));
		
		// Get the four surrounding pixels
		uint8_t p00 = *iv.data(x0, y0);
		uint8_t p10 = *iv.data(x1, y0);
		uint8_t p01 = *iv.data(x0, y1);
		uint8_t p11 = *iv.data(x1, y1);
		
		// Bilinear interpolation
		float val = p00 * (1 - fx) * (1 - fy) +
					p10 * fx * (1 - fy) +
					p01 * (1 - fx) * fy +
					p11 * fx * fy;
		
		return static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
	}

	/**
	 * Sample a line at a given angle through a point, returning luminance values.
	 * @param iv The image to sample from
	 * @param centerX Center X coordinate of the line
	 * @param centerY Center Y coordinate of the line  
	 * @param angleDegrees Angle in degrees (0 = horizontal, 90 = vertical)
	 * @param length Total length of the line to sample
	 * @param result Output vector of sampled luminance values
	 */
	static void SampleLineAtAngle(const ImageView& iv, float centerX, float centerY,
								  float angleDegrees, int length, std::vector<uint8_t>& result)
	{
		result.clear();
		result.reserve(length);
		
		float angleRad = angleDegrees * 3.14159265358979f / 180.0f;
		float dx = std::cos(angleRad);
		float dy = std::sin(angleRad);
		
		// Start from half-length before center
		float startX = centerX - (length / 2.0f) * dx;
		float startY = centerY - (length / 2.0f) * dy;
		
		for (int i = 0; i < length; ++i) {
			float x = startX + i * dx;
			float y = startY + i * dy;
			
			// Check bounds
			if (x < 0 || x >= iv.width() - 1 || y < 0 || y >= iv.height() - 1) {
				result.push_back(255); // White for out of bounds
			} else {
				result.push_back(BilinearInterpolate(iv, x, y));
			}
		}
	}

	/**
	 * Get a full row at a specific angle with interpolation, suitable for barcode decoding.
	 * Now supports ALL angles from 0° to 90° including vertical scanning.
	 * @param iv The image to sample from
	 * @param rowY The Y coordinate (center of the angled line)
	 * @param angleDegrees Angle in degrees from horizontal (-90 to +90)
	 * @param result Output pattern row
	 * @param threshold Binarization threshold
	 * @return true if successful
	 */
	static bool GetAngledPatternRow(const ImageView& iv, int rowY, float angleDegrees,
									PatternRow& result, int threshold)
	{
		float angleRad = angleDegrees * 3.14159265358979f / 180.0f;
		float cosA = std::cos(angleRad);
		float sinA = std::sin(angleRad);
		
		// Calculate the effective scan length based on angle
		// For horizontal (0°): scan along width
		// For vertical (90°): scan along height
		// For in-between: scan along the diagonal
		float effectiveLength;
		if (std::abs(cosA) < 0.01f) {
			// Nearly vertical - scan along height
			effectiveLength = static_cast<float>(iv.height());
		} else if (std::abs(sinA) < 0.01f) {
			// Nearly horizontal - scan along width
			effectiveLength = static_cast<float>(iv.width());
		} else {
			// Angled - use the longer of width/cos or height/sin
			effectiveLength = std::min(
				static_cast<float>(iv.width()) / std::abs(cosA),
				static_cast<float>(iv.height()) / std::abs(sinA)
			);
		}
		
		int sampleLength = static_cast<int>(effectiveLength);
		if (sampleLength < 20)
			return false;
		
		std::vector<uint8_t> samples;
		
		// For vertical or near-vertical scanning, use column as starting point
		float centerX, centerY;
		if (std::abs(angleDegrees) > 80.0f) {
			// Near-vertical: use rowY as X position, scan vertically
			centerX = static_cast<float>(std::min(rowY, iv.width() - 1));
			centerY = iv.height() / 2.0f;
		} else {
			// Normal: use rowY as Y position
			centerX = iv.width() / 2.0f;
			centerY = static_cast<float>(rowY);
		}
		
		SampleLineAtAngle(iv, centerX, centerY, angleDegrees, sampleLength, samples);
		
		if (samples.size() < 10)
			return false;
		
		// Binarize and convert to pattern row
		result.clear();
		result.reserve(samples.size() / 2);
		
		bool lastBlack = samples[0] <= threshold;
		int count = 1;
		
		for (size_t i = 1; i < samples.size(); ++i) {
			bool isBlack = samples[i] <= threshold;
			if (isBlack == lastBlack) {
				++count;
			} else {
				result.push_back(static_cast<uint16_t>(count));
				lastBlack = isBlack;
				count = 1;
			}
		}
		result.push_back(static_cast<uint16_t>(count)); // Last run
		
		// Ensure we start with a space (white)
		if (samples[0] <= threshold && !result.empty()) {
			result.insert(result.begin(), 0);
		}
		
		return result.size() >= 4;
	}

	/**
	 * Sample a line with perspective correction (for barcodes angled in Z-axis).
	 * When a barcode is tilted toward/away from the camera, one side appears compressed.
	 * This function applies a non-linear sampling to compensate.
	 * 
	 * @param iv The image to sample from
	 * @param centerX Center X of the line
	 * @param centerY Center Y of the line
	 * @param length Length to sample
	 * @param perspectiveFactor Factor for perspective (0 = none, positive = right side closer, negative = left side closer)
	 * @param result Output samples
	 */
	static void SampleLineWithPerspective(const ImageView& iv, float centerX, float centerY,
										  int length, float perspectiveFactor, std::vector<uint8_t>& result)
	{
		result.clear();
		result.reserve(length);
		
		float startX = centerX - length / 2.0f;
		
		for (int i = 0; i < length; ++i) {
			// Apply perspective warping: t goes from 0 to 1
			float t = static_cast<float>(i) / length;
			
			// Non-linear mapping based on perspective factor
			// perspectiveFactor > 0: compress left, expand right (right side closer)
			// perspectiveFactor < 0: expand left, compress right (left side closer)
			float warpedT;
			if (std::abs(perspectiveFactor) < 0.01f) {
				warpedT = t;
			} else {
				// Use a power function for perspective warping
				float power = 1.0f + perspectiveFactor;
				warpedT = std::pow(t, power);
			}
			
			float x = startX + warpedT * length;
			float y = centerY;
			
			if (x < 0 || x >= iv.width() - 1 || y < 0 || y >= iv.height() - 1) {
				result.push_back(255);
			} else {
				result.push_back(BilinearInterpolate(iv, x, y));
			}
		}
	}

	/**
	 * Get a pattern row with perspective compensation.
	 * Tries multiple perspective factors to find a readable barcode.
	 */
	static bool GetPerspectivePatternRow(const ImageView& iv, int rowY, PatternRow& result, 
										 int threshold, float perspectiveFactor)
	{
		std::vector<uint8_t> samples;
		SampleLineWithPerspective(iv, iv.width() / 2.0f, static_cast<float>(rowY), 
								  iv.width(), perspectiveFactor, samples);
		
		if (samples.size() < 10)
			return false;
		
		result.clear();
		result.reserve(samples.size() / 2);
		
		bool lastBlack = samples[0] <= threshold;
		int count = 1;
		
		for (size_t i = 1; i < samples.size(); ++i) {
			bool isBlack = samples[i] <= threshold;
			if (isBlack == lastBlack) {
				++count;
			} else {
				result.push_back(static_cast<uint16_t>(count));
				lastBlack = isBlack;
				count = 1;
			}
		}
		result.push_back(static_cast<uint16_t>(count));
		
		if (samples[0] <= threshold && !result.empty()) {
			result.insert(result.begin(), 0);
		}
		
		return result.size() >= 4;
	}

	/**
	 * Get perspective factors to try for Z-axis angle correction.
	 * Returns factors representing different amounts of perspective distortion.
	 */
	static std::vector<float> GetPerspectiveFactors()
	{
		// Try several perspective corrections: none, mild, moderate
		// Positive = right side closer, Negative = left side closer
		return {0.0f, 0.3f, -0.3f, 0.5f, -0.5f, 0.7f, -0.7f};
	}

	/**
	 * Estimate the black point (threshold) for a line sample using histogram analysis.
	 */
	static int EstimateThreshold(const std::vector<uint8_t>& samples)
	{
		if (samples.empty())
			return 127;
		
		// Simple histogram-based threshold estimation
		std::array<int, 256> histogram = {};
		for (uint8_t s : samples)
			histogram[s]++;
		
		// Find peaks
		int darkPeak = 0, darkCount = 0;
		int lightPeak = 255, lightCount = 0;
		
		for (int i = 0; i < 128; ++i) {
			if (histogram[i] > darkCount) {
				darkCount = histogram[i];
				darkPeak = i;
			}
		}
		for (int i = 128; i < 256; ++i) {
			if (histogram[i] > lightCount) {
				lightCount = histogram[i];
				lightPeak = i;
			}
		}
		
		// Threshold is between the two peaks
		return (darkPeak + lightPeak) / 2;
	}

	/**
	 * Get the list of angles to try for angled scanning.
	 * Returns a small set of key angles for fast real-time performance.
	 * Covers horizontal, common tilts, and vertical orientations.
	 */
	static std::vector<float> GetScanAngles(bool includeHorizontal = false)
	{
		std::vector<float> angles;
		if (includeHorizontal)
			angles.push_back(0.0f);
		
		// Only try a few key angles for speed (camera needs to be fast)
		// Covers: slight tilt, moderate tilt, steep angle, and vertical
		for (float angle : {15.0f, 45.0f, 75.0f, 90.0f}) {
			angles.push_back(angle);
			angles.push_back(-angle);
		}
		return angles;
	}

	/**
	 * Transform a point from angled scan coordinates back to image coordinates.
	 * When scanning at an angle, the x-coordinate along the scan line needs to be
	 * transformed back to the original image coordinate system.
	 * 
	 * @param scanX X position along the angled scan line
	 * @param rowY The row Y coordinate (center of the angled line)
	 * @param angleDegrees The angle used for scanning
	 * @param imageWidth Width of the original image
	 * @return The point in image coordinates
	 */
	static PointI TransformFromScanCoords(int scanX, int rowY, float angleDegrees, int imageWidth)
	{
		float angleRad = angleDegrees * 3.14159265358979f / 180.0f;
		float cosA = std::cos(angleRad);
		float sinA = std::sin(angleRad);
		
		// Calculate effective scan width at this angle
		float effectiveWidth = static_cast<float>(imageWidth) / std::abs(cosA);
		
		// scanX is position along the angled line (0 to effectiveWidth)
		// The line starts at the left edge of the image at the given rowY
		// and extends at angleDegrees
		
		float centerX = imageWidth / 2.0f;
		float centerY = static_cast<float>(rowY);
		
		// Convert scanX to position relative to center of scan line
		float offsetFromCenter = scanX - effectiveWidth / 2.0f;
		
		// Transform to image coordinates
		float imgX = centerX + offsetFromCenter * cosA;
		float imgY = centerY + offsetFromCenter * sinA;
		
		return PointI{static_cast<int>(imgX), static_cast<int>(imgY)};
	}

	/**
	 * Transform a barcode position (4 corners) from scan coordinates to image coordinates.
	 * Handles ALL angles including vertical (90°).
	 * @param xStart Start X position in scan coordinates
	 * @param xStop Stop X position in scan coordinates
	 * @param scanPos The scan position (row Y for horizontal, column X for vertical)
	 * @param angleDegrees The angle used for scanning
	 * @param imageWidth Width of the original image
	 * @param imageHeight Height of the original image (needed for vertical)
	 * @param barcodeHeight Approximate height for the barcode bounding box
	 * @return Quadrilateral position in image coordinates
	 */
	static QuadrilateralI TransformPosition(int xStart, int xStop, int scanPos, float angleDegrees, 
											int imageWidth, int barcodeHeight = 20, int imageHeight = 0)
	{
		float angleRad = angleDegrees * 3.14159265358979f / 180.0f;
		float cosA = std::cos(angleRad);
		float sinA = std::sin(angleRad);
		
		bool isNearVertical = std::abs(angleDegrees) > 80.0f;
		
		// Calculate effective scan length
		float effectiveLength;
		if (std::abs(cosA) < 0.01f) {
			effectiveLength = static_cast<float>(imageHeight > 0 ? imageHeight : imageWidth);
		} else if (std::abs(sinA) < 0.01f) {
			effectiveLength = static_cast<float>(imageWidth);
		} else {
			effectiveLength = static_cast<float>(imageWidth) / (std::abs(cosA) > 0.01f ? std::abs(cosA) : 0.01f);
		}
		
		float centerX, centerY;
		if (isNearVertical) {
			centerX = static_cast<float>(scanPos);
			centerY = (imageHeight > 0 ? imageHeight : imageWidth) / 2.0f;
		} else {
			centerX = imageWidth / 2.0f;
			centerY = static_cast<float>(scanPos);
		}
		
		// Convert scan positions to offsets from center
		float startOffset = xStart - effectiveLength / 2.0f;
		float stopOffset = xStop - effectiveLength / 2.0f;
		
		// Calculate the four corners
		// The barcode extends perpendicular to the scan direction
		float perpX = -sinA;  // Perpendicular direction X
		float perpY = cosA;   // Perpendicular direction Y
		float halfHeight = barcodeHeight / 2.0f;
		
		// Start point on scan line
		float startImgX = centerX + startOffset * cosA;
		float startImgY = centerY + startOffset * sinA;
		
		// Stop point on scan line
		float stopImgX = centerX + stopOffset * cosA;
		float stopImgY = centerY + stopOffset * sinA;
		
		// Four corners: top-left, top-right, bottom-right, bottom-left
		PointI topLeft{static_cast<int>(startImgX + perpX * halfHeight), 
					   static_cast<int>(startImgY + perpY * halfHeight)};
		PointI topRight{static_cast<int>(stopImgX + perpX * halfHeight), 
						static_cast<int>(stopImgY + perpY * halfHeight)};
		PointI bottomRight{static_cast<int>(stopImgX - perpX * halfHeight), 
						   static_cast<int>(stopImgY - perpY * halfHeight)};
		PointI bottomLeft{static_cast<int>(startImgX - perpX * halfHeight), 
						  static_cast<int>(startImgY - perpY * halfHeight)};
		
		return QuadrilateralI{topLeft, topRight, bottomRight, bottomLeft};
	}
};

} // namespace ZXing::OneD
