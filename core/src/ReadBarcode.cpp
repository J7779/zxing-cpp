/*
* Copyright 2019 Axel Waggershauser
*/
// SPDX-License-Identifier: Apache-2.0

#include "ReadBarcode.h"

#if !defined(ZXING_READERS) && !defined(ZXING_WRITERS)
#include "Version.h"
#endif

#ifdef ZXING_READERS
#include "GlobalHistogramBinarizer.h"
#include "HybridBinarizer.h"
#include "MultiFormatReader.h"
#include "Pattern.h"
#include "ThresholdBinarizer.h"
#endif

#include <climits>
#include <memory>
#include <stdexcept>

namespace ZXing {

#ifdef ZXING_READERS

class LumImage : public Image
{
public:
	using Image::Image;

	uint8_t* data() { return const_cast<uint8_t*>(Image::data()); }
	const uint8_t* data() const { return Image::data(); }
};

template<typename P>
static LumImage ExtractLum(const ImageView& iv, P projection)
{
	LumImage res(iv.width(), iv.height());

	auto* dst = res.data();
	for(int y = 0; y < iv.height(); ++y)
		for(int x = 0, w = iv.width(); x < w; ++x)
			*dst++ = projection(iv.data(x, y));

	return res;
}

/**
 * ENHANCED: Upscale an image using bilinear interpolation for better 1D barcode detection at distance.
 * When barcodes are too small, the narrow bars may be only 1-2 pixels wide, leading to detection failures.
 * Upscaling with interpolation creates smoother transitions that are easier to threshold correctly.
 */
class LumImageUpscaler
{
	LumImage upscaled;
	
public:
	ImageView layer;
	
	LumImageUpscaler(const ImageView& iv, int factor)
	{
		if (factor < 2 || factor > 4)
			return;
		
		int newWidth = iv.width() * factor;
		int newHeight = iv.height() * factor;
		
		upscaled = LumImage(newWidth, newHeight);
		auto* dst = upscaled.data();
		
		// Bilinear interpolation upscaling
		for (int dy = 0; dy < newHeight; ++dy) {
			float srcY = static_cast<float>(dy) / factor;
			int y0 = static_cast<int>(srcY);
			int y1 = std::min(y0 + 1, iv.height() - 1);
			float fy = srcY - y0;
			
			for (int dx = 0; dx < newWidth; ++dx) {
				float srcX = static_cast<float>(dx) / factor;
				int x0 = static_cast<int>(srcX);
				int x1 = std::min(x0 + 1, iv.width() - 1);
				float fx = srcX - x0;
				
				// Get 4 surrounding pixels
				uint8_t p00 = *iv.data(x0, y0);
				uint8_t p10 = *iv.data(x1, y0);
				uint8_t p01 = *iv.data(x0, y1);
				uint8_t p11 = *iv.data(x1, y1);
				
				// Bilinear interpolation
				float val = p00 * (1 - fx) * (1 - fy) +
							p10 * fx * (1 - fy) +
							p01 * (1 - fx) * fy +
							p11 * fx * fy;
				
				*dst++ = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
			}
		}
		
		layer = upscaled;
	}
	
	bool isValid() const { return upscaled.width() > 0; }
};

class LumImagePyramid
{
	std::vector<LumImage> buffers;

	template<int N>
	void addLayer()
	{
		auto siv = layers.back();
		buffers.emplace_back(siv.width() / N, siv.height() / N);
		layers.push_back(buffers.back());
		auto& div = buffers.back();
		auto* d   = div.data();

		for (int dy = 0; dy < div.height(); ++dy)
			for (int dx = 0; dx < div.width(); ++dx) {
				int sum = (N * N) / 2;
				for (int ty = 0; ty < N; ++ty)
					for (int tx = 0; tx < N; ++tx)
						sum += *siv.data(dx * N + tx, dy * N + ty);
				*d++ = sum / (N * N);
			}
	}

	void addLayer(int factor)
	{
		// help the compiler's auto-vectorizer by hard-coding the scale factor
		switch (factor) {
		case 2: addLayer<2>(); break;
		case 3: addLayer<3>(); break;
		case 4: addLayer<4>(); break;
		default: throw std::invalid_argument("Invalid ReaderOptions::downscaleFactor"); break;
		}
	}

public:
	std::vector<ImageView> layers;

	LumImagePyramid(const ImageView& iv, int threshold, int factor)
	{
		if (factor < 2)
			throw std::invalid_argument("Invalid ReaderOptions::downscaleFactor");

		layers.push_back(iv);
		// TODO: if only matrix codes were considered, then using std::min would be sufficient (see #425)
		while (threshold > 0 && std::max(layers.back().width(), layers.back().height()) > threshold &&
			   std::min(layers.back().width(), layers.back().height()) >= factor)
			addLayer(factor);
#if 0
		// Reversing the layers means we'd start with the smallest. that can make sense if we are only looking for a
		// single symbol. If we start with the higher resolution, we get better (high res) position information.
		// TODO: see if masking out higher res layers based on found symbols in lower res helps overall performance.
		std::reverse(layers.begin(), layers.end());
#endif
	}
};

ImageView SetupLumImageView(ImageView iv, LumImage& lum, const ReaderOptions& opts)
{
	if (iv.format() == ImageFormat::None)
		throw std::invalid_argument("Invalid image format");

	if (opts.binarizer() == Binarizer::GlobalHistogram || opts.binarizer() == Binarizer::LocalAverage) {
		// manually spell out the 3 most common pixel formats to get at least gcc to vectorize the code
		if (iv.format() == ImageFormat::RGB && iv.pixStride() == 3) {
			lum = ExtractLum(iv, [](const uint8_t* src) { return RGBToLum(src[0], src[1], src[2]); });
		} else if (iv.format() == ImageFormat::RGBA && iv.pixStride() == 4) {
			lum = ExtractLum(iv, [](const uint8_t* src) { return RGBToLum(src[0], src[1], src[2]); });
		} else if (iv.format() == ImageFormat::BGR && iv.pixStride() == 3) {
			lum = ExtractLum(iv, [](const uint8_t* src) { return RGBToLum(src[2], src[1], src[0]); });
		} else if (iv.format() != ImageFormat::Lum) {
			lum = ExtractLum(iv, [r = RedIndex(iv.format()), g = GreenIndex(iv.format()), b = BlueIndex(iv.format())](
									 const uint8_t* src) { return RGBToLum(src[r], src[g], src[b]); });
		} else if (iv.pixStride() != 1) {
			// GlobalHistogram and LocalAverage need dense line memory layout
			lum = ExtractLum(iv, [](const uint8_t* src) { return *src; });
		}
		if (lum.data())
			return lum;
	}
	return iv;
}

std::unique_ptr<BinaryBitmap> CreateBitmap(ZXing::Binarizer binarizer, const ImageView& iv)
{
	switch (binarizer) {
	case Binarizer::BoolCast: return std::make_unique<ThresholdBinarizer>(iv, 0);
	case Binarizer::FixedThreshold: return std::make_unique<ThresholdBinarizer>(iv, 127);
	case Binarizer::GlobalHistogram: return std::make_unique<GlobalHistogramBinarizer>(iv);
	case Binarizer::LocalAverage: return std::make_unique<HybridBinarizer>(iv);
	}
	return {}; // silence gcc warning
}

Barcode ReadBarcode(const ImageView& _iv, const ReaderOptions& opts)
{
	return FirstOrDefault(ReadBarcodes(_iv, ReaderOptions(opts).setMaxNumberOfSymbols(1)));
}

Barcodes ReadBarcodes(const ImageView& _iv, const ReaderOptions& opts)
{
	if (sizeof(PatternType) < 4 && (_iv.width() > 0xffff || _iv.height() > 0xffff))
		throw std::invalid_argument("Maximum image width/height is 65535");

	if (!_iv.data() || _iv.width() * _iv.height() == 0)
		throw std::invalid_argument("ImageView is null/empty");

	LumImage lum;
	ImageView iv = SetupLumImageView(_iv, lum, opts);
	MultiFormatReader reader(opts);

	if (opts.isPure())
		return {reader.read(*CreateBitmap(opts.binarizer(), iv)).setReaderOptions(opts)};

	std::unique_ptr<MultiFormatReader> closedReader;
#ifdef ZXING_EXPERIMENTAL_API
	auto formatsBenefittingFromClosing = BarcodeFormat::Aztec | BarcodeFormat::DataMatrix | BarcodeFormat::QRCode | BarcodeFormat::MicroQRCode;
	ReaderOptions closedOptions = opts;
	if (opts.tryDenoise() && opts.hasFormat(formatsBenefittingFromClosing) && _iv.height() >= 3) {
		closedOptions.setFormats((opts.formats().empty() ? BarcodeFormat::Any : opts.formats()) & formatsBenefittingFromClosing);
		closedReader = std::make_unique<MultiFormatReader>(closedOptions);
	}
#endif
	LumImagePyramid pyramid(iv, opts.downscaleThreshold() * opts.tryDownscale(), opts.downscaleFactor());

	Barcodes res;
	int maxSymbols = opts.maxNumberOfSymbols() ? opts.maxNumberOfSymbols() : INT_MAX;
	for (auto&& iv : pyramid.layers) {
		auto bitmap = CreateBitmap(opts.binarizer(), iv);
		for (int close = 0; close <= (closedReader ? 1 : 0); ++close) {
			if (close) {
				// if we already inverted the image in the first round, we need to undo that first
				if (bitmap->inverted())
					bitmap->invert();
				bitmap->close();
			}

			// TODO: check if closing after invert would be beneficial
			for (int invert = 0; invert <= static_cast<int>(opts.tryInvert() && !close); ++invert) {
				if (invert)
					bitmap->invert();
				auto rs = (close ? *closedReader : reader).readMultiple(*bitmap, maxSymbols);
				for (auto& r : rs) {
					if (iv.width() != _iv.width())
						r.setPosition(Scale(r.position(), _iv.width() / iv.width()));
					if (!Contains(res, r)) {
						r.setReaderOptions(opts);
						r.setIsInverted(bitmap->inverted());
						res.push_back(std::move(r));
						--maxSymbols;
					}
				}
				if (maxSymbols <= 0)
					return res;
			}
		}
	}

	// ENHANCED: Try upscaled image if no results found and upscaling is enabled
	// This helps with barcodes that are far away where bars are only 1-2 pixels wide
	if (res.empty() && opts.tryUpscale()) {
		// Try multiple upscale factors - start with 2x, then 3x, then 4x if needed
		// 4x is aggressive but helps with very distant barcodes
		for (int factor : {2, 3, 4}) {
			LumImageUpscaler upscaler(iv, factor);
			if (!upscaler.isValid())
				continue;
				
			auto upscaledBitmap = CreateBitmap(opts.binarizer(), upscaler.layer);
			for (int invert = 0; invert <= static_cast<int>(opts.tryInvert()); ++invert) {
				if (invert)
					upscaledBitmap->invert();
				auto rs = reader.readMultiple(*upscaledBitmap, maxSymbols);
				for (auto& r : rs) {
					// Scale position back to original image size
					auto pos = r.position();
					r.setPosition({pos[0] / factor, pos[1] / factor, pos[2] / factor, pos[3] / factor});
					if (!Contains(res, r)) {
						r.setReaderOptions(opts);
						r.setIsInverted(upscaledBitmap->inverted());
						res.push_back(std::move(r));
						--maxSymbols;
					}
				}
				if (maxSymbols <= 0)
					return res;
			}
			
			// If we found results with this factor, don't try larger ones
			if (!res.empty())
				break;
		}
	}

	return res;
}

#else // ZXING_READERS

Barcode ReadBarcode(const ImageView&, const ReaderOptions&)
{
	throw std::runtime_error("This build of zxing-cpp does not support reading barcodes.");
}

Barcodes ReadBarcodes(const ImageView&, const ReaderOptions&)
{
	throw std::runtime_error("This build of zxing-cpp does not support reading barcodes.");
}

#endif // ZXING_READERS

} // ZXing
