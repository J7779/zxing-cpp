/*
* Copyright 2016 Nu-book Inc.
* Copyright 2016 ZXing authors
* Copyright 2020 Axel Waggershauser
*/
// SPDX-License-Identifier: Apache-2.0

#include "ODReader.h"

#include "BinaryBitmap.h"
#include "ReaderOptions.h"
#include "ODAngledScanning.h"
#include "ODCodabarReader.h"
#include "ODCode128Reader.h"
#include "ODCode39Reader.h"
#include "ODCode93Reader.h"
#include "ODDataBarExpandedReader.h"
#include "ODDataBarLimitedReader.h"
#include "ODDataBarReader.h"
#include "ODDXFilmEdgeReader.h"
#include "ODITFReader.h"
#include "ODMultiUPCEANReader.h"
#include "Barcode.h"

#include "ODDiagnostics.h"

#include <algorithm>
#include <cmath>
#include <utility>

#ifdef PRINT_DEBUG
#include "BitMatrix.h"
#include "BitMatrixIO.h"
#endif

namespace ZXing {

void IncrementLineCount(Barcode& r)
{
	++r._lineCount;
}

} // namespace ZXing

namespace ZXing::OneD {

Reader::Reader(const ReaderOptions& opts) : ZXing::Reader(opts)
{
	_readers.reserve(8);

	auto formats = opts.formats().empty() ? BarcodeFormat::Any : opts.formats();

	if (formats.testFlags(BarcodeFormat::EAN13 | BarcodeFormat::UPCA | BarcodeFormat::EAN8 | BarcodeFormat::UPCE))
		_readers.emplace_back(new MultiUPCEANReader(opts));

	if (formats.testFlag(BarcodeFormat::Code39))
		_readers.emplace_back(new Code39Reader(opts));
	if (formats.testFlag(BarcodeFormat::Code93))
		_readers.emplace_back(new Code93Reader(opts));
	if (formats.testFlag(BarcodeFormat::Code128))
		_readers.emplace_back(new Code128Reader(opts));
	if (formats.testFlag(BarcodeFormat::ITF))
		_readers.emplace_back(new ITFReader(opts));
	if (formats.testFlag(BarcodeFormat::Codabar))
		_readers.emplace_back(new CodabarReader(opts));
	if (formats.testFlag(BarcodeFormat::DataBar))
		_readers.emplace_back(new DataBarReader(opts));
	if (formats.testFlag(BarcodeFormat::DataBarExpanded))
		_readers.emplace_back(new DataBarExpandedReader(opts));
	if (formats.testFlag(BarcodeFormat::DataBarLimited))
		_readers.emplace_back(new DataBarLimitedReader(opts));
	if (formats.testFlag(BarcodeFormat::DXFilmEdge))
		_readers.emplace_back(new DXFilmEdgeReader(opts));
}

Reader::~Reader() = default;

/**
* We're going to examine rows from the middle outward, searching alternately above and below the
* middle, and farther out each time. rowStep is the number of rows between each successive
* attempt above and below the middle. So we'd scan row middle, then middle - rowStep, then
* middle + rowStep, then middle - (2 * rowStep), etc.
* rowStep is bigger as the image is taller, but is always at least 1. We've somewhat arbitrarily
* decided that moving up and down by about 1/16 of the image is pretty good; we try more of the
* image if "trying harder".
* 
* ENHANCED: When tryHarder is enabled, we now scan significantly more rows (up to 512 divisions)
* to increase the chance of finding barcodes that may only be visible in certain scan lines.
*/
static Barcodes DoDecode(const std::vector<std::unique_ptr<RowReader>>& readers, const BinaryBitmap& image, bool tryHarder,
						 bool rotate, bool isPure, int maxSymbols, int minLineCount, bool returnErrors)
{
	Barcodes res;

	std::vector<std::unique_ptr<RowReader::DecodingState>> decodingState(readers.size());

	int width = image.width();
	int height = image.height();

	OD_DIAG("[1D_DECODE] DoDecode START image=" + std::to_string(width) + "x" + std::to_string(height)
		+ " rotate=" + (rotate ? "true" : "false") + " tryHarder=" + (tryHarder ? "true" : "false")
		+ " isPure=" + (isPure ? "true" : "false") + " maxSymbols=" + std::to_string(maxSymbols)
		+ " minLineCount=" + std::to_string(minLineCount) + " readers=" + std::to_string(readers.size()));

	if (rotate)
		std::swap(width, height);

	int middle = height / 2;
	int rowStep = std::max(1, height / ((tryHarder && !isPure) ? (maxSymbols == 1 ? 512 : 1024) : 64));
	int maxLines = tryHarder ?
		height :
		30;

	OD_DIAG("[1D_SCAN_PARAMS] middle=" + std::to_string(middle) + " rowStep=" + std::to_string(rowStep)
		+ " maxLines=" + std::to_string(maxLines) + " dims=" + std::to_string(width) + "x" + std::to_string(height));

	if (isPure)
		minLineCount = 1;
	else
		minLineCount = std::min(minLineCount, height);
	std::vector<int> checkRows;

	PatternRow bars;
	bars.reserve(128);
	int rowsScanned = 0;
	int rowsWithBars = 0;
	int totalDecodeAttempts = 0;
	int totalValidResults = 0;
	int totalInvalidResults = 0;

#ifdef PRINT_DEBUG
	BitMatrix dbg(width, height);
#endif

	for (int i = 0; i < maxLines; i++) {

		// Scanning from the middle out. Determine which row we're looking at next:
		int rowStepsAboveOrBelow = (i + 1) / 2;
		bool isAbove = (i & 0x01) == 0; // i.e. is x even?
		int rowNumber = middle + rowStep * (isAbove ? rowStepsAboveOrBelow : -rowStepsAboveOrBelow);
		bool isCheckRow = false;
		if (rowNumber < 0 || rowNumber >= height) {
			// Oops, if we run off the top or bottom, stop
			break;
		}

		// See if we have additional check rows (see below) to process
		if (checkRows.size()) {
			--i;
			rowNumber = checkRows.back();
			checkRows.pop_back();
			isCheckRow = true;
			if (rowNumber < 0 || rowNumber >= height)
				continue;
		}

		++rowsScanned;
		if (!image.getPatternRow(rowNumber, rotate ? 90 : 0, bars)) {
			OD_DIAG("[1D_ROW#" + std::to_string(rowNumber) + "] getPatternRow failed (empty/threshold)");
			continue;
		}
		++rowsWithBars;
		OD_DIAG("[1D_ROW#" + std::to_string(rowNumber) + "] bars=" + std::to_string(bars.size())
			+ " isCheckRow=" + (isCheckRow ? "true" : "false"));

#ifdef PRINT_DEBUG
		bool val = false;
		int x = 0;
		for (auto b : bars) {
			for(int j = 0; j < b; ++j)
				dbg.set(x++, rowNumber, val);
			val = !val;
		}
#endif

		// While we have the image data in a PatternRow, it's fairly cheap to reverse it in place to
		// handle decoding upside down barcodes.
		// TODO: the DataBarExpanded (stacked) decoder depends on seeing each line from both directions. This is
		// 'surprising' and inconsistent. It also requires the decoderState to be shared between normal and reversed
		// scans, which makes no sense in general because it would mix partial detection data from two codes of the same
		// type next to each other. See also https://github.com/zxing-cpp/zxing-cpp/issues/87
		for (bool upsideDown : {false, true}) {
			// trying again?
			if (upsideDown) {
				// reverse the row and continue
				std::reverse(bars.begin(), bars.end());
			}
			// Look for a barcode
			for (size_t r = 0; r < readers.size(); ++r) {
				if (isPure && i && !decodingState[r])
					continue;

				PatternView next(bars);
				do {
					++totalDecodeAttempts;
					Barcode result = readers[r]->decodePattern(rowNumber, next, decodingState[r]);
					if (result.isValid() || (returnErrors && result.error())) {
						if (result.isValid()) {
							++totalValidResults;
							OD_DIAG("[1D_MATCH] row=" + std::to_string(rowNumber) + " reader=" + std::to_string(r)
								+ " format=" + ToString(result.format()) + " text=" + result.text().substr(0, 40)
								+ " upsideDown=" + (upsideDown ? "true" : "false")
								+ " lineCount=" + std::to_string(result.lineCount()));
						} else {
							++totalInvalidResults;
							OD_DIAG("[1D_ERROR] row=" + std::to_string(rowNumber) + " reader=" + std::to_string(r)
								+ " format=" + ToString(result.format()) + " error=" + ToString(result.error()));
						}
						IncrementLineCount(result);
						if (upsideDown) {
							// update position (flip horizontally).
							auto points = result.position();
							for (auto& p : points) {
								p = {width - p.x - 1, p.y};
							}
							result.setPosition(std::move(points));
						}
						if (rotate) {
							auto points = result.position();
							for (auto& p : points) {
								p = {p.y, width - p.x - 1};
							}
							result.setPosition(std::move(points));
						}

						// check if we know this code already
						for (auto& other : res) {
							if (result == other) {
								// merge the position information
								auto dTop = maxAbsComponent(other.position().topLeft() - result.position().topLeft());
								auto dBot = maxAbsComponent(other.position().bottomLeft() - result.position().topLeft());
								auto points = other.position();
								if (dTop < dBot || (dTop == dBot && rotate ^ (sumAbsComponent(points[0]) >
																			  sumAbsComponent(result.position()[0])))) {
									points[0] = result.position()[0];
									points[1] = result.position()[1];
								} else {
									points[2] = result.position()[2];
									points[3] = result.position()[3];
								}
								other.setPosition(points);
								IncrementLineCount(other);
								// clear the result, so we don't insert it again below
								result = Barcode();
								break;
							}
						}

						if (result.format() != BarcodeFormat::None) {
							res.push_back(std::move(result));

							// if we found a valid code we have not seen before but a minLineCount > 1,
							// add additional check rows above and below the current one
							if (!isCheckRow && minLineCount > 1 && rowStep > 1) {
								checkRows = {rowNumber - 1, rowNumber + 1};
								if (rowStep > 2)
									checkRows.insert(checkRows.end(), {rowNumber - 2, rowNumber + 2});
							}
						}

						if (maxSymbols && Reduce(res, 0, [&](int s, const Barcode& r) {
											  return s + (r.lineCount() >= minLineCount);
										  }) == maxSymbols) {
							goto out;
						}
					}
					// make sure we make progress and we start the next try on a bar
					next.shift(2 - (next.index() % 2));
					next.extend();
				} while (tryHarder && next.size());
			}
		}
	}

out:
	OD_DIAG("[1D_DECODE_SUMMARY] rowsScanned=" + std::to_string(rowsScanned)
		+ " rowsWithBars=" + std::to_string(rowsWithBars)
		+ " decodeAttempts=" + std::to_string(totalDecodeAttempts)
		+ " validHits=" + std::to_string(totalValidResults)
		+ " invalidHits=" + std::to_string(totalInvalidResults)
		+ " rawResults=" + std::to_string(res.size()));

	// remove all symbols with insufficient line count
#ifdef __cpp_lib_erase_if
	std::erase_if(res, [&](auto&& r) { return r.lineCount() < minLineCount; });
#else
	auto it = std::remove_if(res.begin(), res.end(), [&](auto&& r) { return r.lineCount() < minLineCount; });
	res.erase(it, res.end());
#endif

	// if symbols overlap, remove the one with a lower line count
	for (auto a = res.begin(); a != res.end(); ++a)
		for (auto b = std::next(a); b != res.end(); ++b)
			if (HaveIntersectingBoundingBoxes(a->position(), b->position()))
				*(a->lineCount() < b->lineCount() ? a : b) = Barcode();

#ifdef __cpp_lib_erase_if
	std::erase_if(res, [](auto&& r) { return r.format() == BarcodeFormat::None; });
#else
	it = std::remove_if(res.begin(), res.end(), [](auto&& r) { return r.format() == BarcodeFormat::None; });
	res.erase(it, res.end());
#endif

#ifdef PRINT_DEBUG
	SaveAsPBM(dbg, rotate ? "od-log-r.pnm" : "od-log.pnm");
#endif

	return res;
}

/**
 * Decode at a specific angle using interpolated scanning.
 * This helps detect barcodes that are not perfectly horizontal.
 * Supports ALL angles from 0° to 90° including fully vertical barcodes.
 */
static Barcodes DoDecodeAngled(const std::vector<std::unique_ptr<RowReader>>& readers, const BinaryBitmap& image,
							   float angleDegrees, bool tryHarder, int maxSymbols, int minLineCount, bool returnErrors)
{
	Barcodes res;
	
	const ImageView& buffer = image.buffer();
	if (buffer.format() == ImageFormat::None || buffer.data(0, 0) == nullptr)
		return res;
	
	int width = image.width();
	int height = image.height();
	
	// For near-vertical angles, we scan "columns" instead of "rows"
	bool isNearVertical = std::abs(angleDegrees) > 80.0f;
	int scanDimension = isNearVertical ? width : height;
	
	int middle = scanDimension / 2;
	int step = std::max(1, scanDimension / (tryHarder ? 128 : 32));
	int maxLines = tryHarder ? scanDimension / step : 20;
	
	std::vector<std::unique_ptr<RowReader::DecodingState>> decodingState(readers.size());
	PatternRow bars;
	bars.reserve(128);
	
	for (int i = 0; i < maxLines; ++i) {
		int stepsAboveOrBelow = (i + 1) / 2;
		bool isAbove = (i & 0x01) == 0;
		int scanPos = middle + step * (isAbove ? stepsAboveOrBelow : -stepsAboveOrBelow);
		
		if (scanPos < 0 || scanPos >= scanDimension)
			break;
		
		// Sample the line first to estimate threshold adaptively
		float angleRad = angleDegrees * 3.14159265358979f / 180.0f;
		float cosA = std::cos(angleRad);
		float sinA = std::sin(angleRad);
		
		// Calculate effective scan length
		float effectiveLength;
		if (std::abs(cosA) < 0.01f) {
			effectiveLength = static_cast<float>(height);
		} else if (std::abs(sinA) < 0.01f) {
			effectiveLength = static_cast<float>(width);
		} else {
			effectiveLength = std::min(
				static_cast<float>(width) / std::abs(cosA),
				static_cast<float>(height) / std::abs(sinA)
			);
		}
		
		int sampleLength = static_cast<int>(effectiveLength);
		std::vector<uint8_t> samples;
		
		float centerX, centerY;
		if (isNearVertical) {
			centerX = static_cast<float>(std::min(scanPos, width - 1));
			centerY = height / 2.0f;
		} else {
			centerX = width / 2.0f;
			centerY = static_cast<float>(scanPos);
		}
		
		AngledScanner::SampleLineAtAngle(buffer, centerX, centerY, angleDegrees, sampleLength, samples);
		
		// Use adaptive threshold estimation
		int threshold = AngledScanner::EstimateThreshold(samples);
		
		if (!AngledScanner::GetAngledPatternRow(buffer, scanPos, angleDegrees, bars, threshold))
			continue;
		
		if (bars.size() < 10)
			continue;
		
		// Try both directions
		for (bool upsideDown : {false, true}) {
			if (upsideDown)
				std::reverse(bars.begin(), bars.end());
			
			for (size_t r = 0; r < readers.size(); ++r) {
				PatternView next(bars);
				do {
					Barcode result = readers[r]->decodePattern(scanPos, next, decodingState[r]);
					if (result.isValid() || (returnErrors && result.error())) {
						IncrementLineCount(result);
						
						// Transform position from angled scan coordinates to image coordinates
						auto oldPos = result.position();
						int xStart = oldPos.topLeft().x;
						int xStop = oldPos.topRight().x;
						int barcodeHeight = std::max(20, std::abs(oldPos.bottomLeft().y - oldPos.topLeft().y));
						auto newPos = AngledScanner::TransformPosition(
							xStart, xStop, scanPos, angleDegrees, 
							width, barcodeHeight, height);
						result.setPosition(newPos);
						
						bool isDuplicate = false;
						for (auto& other : res) {
							if (result == other) {
								IncrementLineCount(other);
								isDuplicate = true;
								break;
							}
						}
						
						if (!isDuplicate && result.format() != BarcodeFormat::None) {
							res.push_back(std::move(result));
							
							if (maxSymbols && static_cast<int>(res.size()) >= maxSymbols)
								return res;
						}
					}
					next.shift(2 - (next.index() % 2));
					next.extend();
				} while (tryHarder && next.size());
			}
		}
	}
	
	// Filter by line count
#ifdef __cpp_lib_erase_if
	std::erase_if(res, [&](auto&& r) { return r.lineCount() < minLineCount; });
#else
	auto it = std::remove_if(res.begin(), res.end(), [&](auto&& r) { return r.lineCount() < minLineCount; });
	res.erase(it, res.end());
#endif
	
	return res;
}

/**
 * Decode with perspective correction for Z-axis tilted barcodes.
 * When a barcode is tilted toward/away from camera (like on a curved bottle),
 * the bars appear non-uniformly spaced. This tries multiple perspective corrections.
 */
static Barcodes DoDecodePerspective(const std::vector<std::unique_ptr<RowReader>>& readers, const BinaryBitmap& image,
									bool tryHarder, int maxSymbols, int minLineCount, bool returnErrors)
{
	Barcodes res;
	
	const ImageView& buffer = image.buffer();
	if (buffer.format() == ImageFormat::None || buffer.data(0, 0) == nullptr)
		return res;
	
	int width = image.width();
	int height = image.height();
	
	int middle = height / 2;
	int step = std::max(1, height / (tryHarder ? 64 : 16));
	int maxLines = tryHarder ? 32 : 10;
	
	std::vector<std::unique_ptr<RowReader::DecodingState>> decodingState(readers.size());
	PatternRow bars;
	bars.reserve(128);
	
	// Get perspective factors to try
	auto perspectiveFactors = AngledScanner::GetPerspectiveFactors();
	
	for (float perspFactor : perspectiveFactors) {
		if (maxSymbols && static_cast<int>(res.size()) >= maxSymbols)
			break;
			
		for (int i = 0; i < maxLines; ++i) {
			int stepsAboveOrBelow = (i + 1) / 2;
			bool isAbove = (i & 0x01) == 0;
			int rowNumber = middle + step * (isAbove ? stepsAboveOrBelow : -stepsAboveOrBelow);
			
			if (rowNumber < 0 || rowNumber >= height)
				continue;
			
			// Sample with perspective to estimate threshold
			std::vector<uint8_t> samples;
			AngledScanner::SampleLineWithPerspective(buffer, width / 2.0f, static_cast<float>(rowNumber),
													 width, perspFactor, samples);
			int threshold = AngledScanner::EstimateThreshold(samples);
			
			if (!AngledScanner::GetPerspectivePatternRow(buffer, rowNumber, bars, threshold, perspFactor))
				continue;
			
			if (bars.size() < 10)
				continue;
			
			// Try both directions
			for (bool upsideDown : {false, true}) {
				if (upsideDown)
					std::reverse(bars.begin(), bars.end());
				
				for (size_t r = 0; r < readers.size(); ++r) {
					PatternView next(bars);
					do {
						Barcode result = readers[r]->decodePattern(rowNumber, next, decodingState[r]);
						if (result.isValid() || (returnErrors && result.error())) {
							IncrementLineCount(result);
							
							bool isDuplicate = false;
							for (auto& other : res) {
								if (result == other) {
									IncrementLineCount(other);
									isDuplicate = true;
									break;
								}
							}
							
							if (!isDuplicate && result.format() != BarcodeFormat::None) {
								res.push_back(std::move(result));
								
								if (maxSymbols && static_cast<int>(res.size()) >= maxSymbols)
									return res;
							}
						}
						next.shift(2 - (next.index() % 2));
						next.extend();
					} while (tryHarder && next.size());
				}
			}
		}
	}
	
	// Filter by line count
#ifdef __cpp_lib_erase_if
	std::erase_if(res, [&](auto&& r) { return r.lineCount() < minLineCount; });
#else
	auto it2 = std::remove_if(res.begin(), res.end(), [&](auto&& r) { return r.lineCount() < minLineCount; });
	res.erase(it2, res.end());
#endif
	
	return res;
}

Barcode Reader::decode(const BinaryBitmap& image) const
{
	OD_DIAG("[1D_READER] decode(single) START image=" + std::to_string(image.width()) + "x" + std::to_string(image.height())
		+ " tryHarder=" + (_opts.tryHarder() ? "true" : "false") + " tryRotate=" + (_opts.tryRotate() ? "true" : "false")
		+ " tryAngledScanning=" + (_opts.tryAngledScanning() ? "true" : "false")
		+ " relaxedLinearTolerance=" + (_opts.relaxedLinearTolerance() ? "true" : "false"));

	auto result =
		DoDecode(_readers, image, _opts.tryHarder(), false, _opts.isPure(), 1, _opts.minLineCount(), _opts.returnErrors());
	
	if (result.empty() && _opts.tryRotate()) {
		OD_DIAG("[1D_READER] trying 90deg rotation");
		result = DoDecode(_readers, image, _opts.tryHarder(), true, _opts.isPure(), 1, _opts.minLineCount(), _opts.returnErrors());
	}

	if (result.empty() && _opts.tryAngledScanning()) {
		auto angles = AngledScanner::GetScanAngles();
		OD_DIAG("[1D_READER] trying angled scanning, " + std::to_string(angles.size()) + " angles");
		for (float angle : angles) {
			OD_DIAG("[1D_ANGLE] trying " + std::to_string(angle) + " deg");
			result = DoDecodeAngled(_readers, image, angle, _opts.tryHarder(), 1, 
									_opts.minLineCount(), _opts.returnErrors());
			if (!result.empty()) {
				OD_DIAG("[1D_ANGLE] SUCCESS at " + std::to_string(angle) + " deg");
				break;
			}
		}
	}
	
	if (result.empty() && _opts.tryAngledScanning()) {
		OD_DIAG("[1D_READER] trying perspective correction");
		result = DoDecodePerspective(_readers, image, _opts.tryHarder(), 1, 
									 _opts.minLineCount(), _opts.returnErrors());
	}

	OD_DIAG("[1D_READER] decode(single) END found=" + std::to_string(result.size()));
	return FirstOrDefault(std::move(result));
}

Barcodes Reader::decode(const BinaryBitmap& image, int maxSymbols) const
{
	OD_DIAG("[1D_READER] decode(multi) START maxSymbols=" + std::to_string(maxSymbols));

	auto resH = DoDecode(_readers, image, _opts.tryHarder(), false, _opts.isPure(), maxSymbols, _opts.minLineCount(),
						 _opts.returnErrors());
	OD_DIAG("[1D_READER] horizontal pass found=" + std::to_string(resH.size()));

	if ((!maxSymbols || Size(resH) < maxSymbols) && _opts.tryRotate()) {
		OD_DIAG("[1D_READER] trying 90deg rotation");
		auto resV = DoDecode(_readers, image, _opts.tryHarder(), true, _opts.isPure(), maxSymbols - Size(resH),
							 _opts.minLineCount(), _opts.returnErrors());
		OD_DIAG("[1D_READER] rotated pass found=" + std::to_string(resV.size()));
		resH.insert(resH.end(), resV.begin(), resV.end());
	}
	
	if ((!maxSymbols || Size(resH) < maxSymbols) && _opts.tryAngledScanning()) {
		auto angles = AngledScanner::GetScanAngles();
		OD_DIAG("[1D_READER] trying angled scanning, " + std::to_string(angles.size()) + " angles");
		for (float angle : angles) {
			OD_DIAG("[1D_ANGLE] trying " + std::to_string(angle) + " deg");
			auto resA = DoDecodeAngled(_readers, image, angle, _opts.tryHarder(), 
									   maxSymbols - Size(resH), _opts.minLineCount(), _opts.returnErrors());
			for (auto& r : resA) {
				if (!Contains(resH, r)) {
					OD_DIAG("[1D_ANGLE] found new symbol at " + std::to_string(angle) + " deg: " + ToString(r.format()));
					resH.push_back(std::move(r));
					if (maxSymbols && Size(resH) >= maxSymbols)
						return resH;
				}
			}
		}
	}
	
	if ((!maxSymbols || Size(resH) < maxSymbols) && _opts.tryAngledScanning()) {
		OD_DIAG("[1D_READER] trying perspective correction");
		auto resP = DoDecodePerspective(_readers, image, _opts.tryHarder(), 
										maxSymbols - Size(resH), _opts.minLineCount(), _opts.returnErrors());
		OD_DIAG("[1D_READER] perspective found=" + std::to_string(resP.size()));
		for (auto& r : resP) {
			if (!Contains(resH, r)) {
				resH.push_back(std::move(r));
				if (maxSymbols && Size(resH) >= maxSymbols)
					return resH;
			}
		}
	}
	
	OD_DIAG("[1D_READER] decode(multi) END totalFound=" + std::to_string(resH.size()));
	return resH;
}

} // namespace ZXing::OneD
