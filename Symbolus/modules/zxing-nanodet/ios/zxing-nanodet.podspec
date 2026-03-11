Pod::Spec.new do |s|
  s.name             = 'zxing-nanodet'
  s.version          = '1.0.0'
  s.summary          = 'VisionCamera frame processor plugin: NanoDet + ZXing barcodes'
  s.homepage         = 'https://github.com/zxing-cpp/zxing-cpp'
  s.license          = { :type => 'Apache-2.0' }
  s.author           = { 'ZXing-CPP' => 'info@zxing-cpp.org' }
  s.platform         = :ios, '14.0'

  # Source files in this pod
  s.source_files = 'ios/**/*.{h,m,mm,swift}'

  # ── ZXing-CPP core ────────────────────────────────────────────────────────
  # Point at the local core source tree relative to this podspec.
  zxing_root = File.join(__dir__, '../../../../core/src')
  s.preserve_paths = "#{zxing_root}/**/*"
  s.xcconfig = {
    'HEADER_SEARCH_PATHS' => "\"#{zxing_root}\" \"#{zxing_root}/onnx\"",
    'OTHER_CPLUSPLUSFLAGS' => '-DZXING_USE_ONNXRUNTIME=1 -DZXING_READERS -DZXING_WITH_1D -DZXING_WITH_QRCODE -DZXING_WITH_DATAMATRIX -DZXING_WITH_AZTEC -DZXING_WITH_PDF417 -DZXING_WITH_MAXICODE',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
  }

  # Include ZXing core source files needed by the plugin
  s.source_files += [
    "#{zxing_root}/**/*.{h,cpp}",
  ]

  # ── ORT (ONNX Runtime) ────────────────────────────────────────────────────
  s.dependency 'onnxruntime-c', '~> 1.18'

  # ── VisionCamera ─────────────────────────────────────────────────────────
  s.dependency 'VisionCamera'
end
