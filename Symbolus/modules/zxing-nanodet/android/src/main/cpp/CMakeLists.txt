# SPDX-License-Identifier: Apache-2.0
# CMakeLists for the Android JNI component of zxing-nanodet.
#
# Architecture: ORT inference runs entirely in Kotlin (Java ORT API).
# This C++ shared library only provides:
#   1. NanoDet pre/post-processing  (letterbox + GFL decode, no ORT C API needed)
#   2. ZXing barcode decoding       (ReadBarcodes on luma crop)
#
# Path from this file to the zxing-cpp repo root:
#   cpp/ -> main/ -> src/ -> android/ -> zxing-nanodet/ -> modules/ -> Symbolus/ -> zxing-cpp/
#   = 7 parent dirs

cmake_minimum_required(VERSION 3.22)
project(zxing_nanodet_jni VERSION "2.3.0")

# ZXing-CPP core path — override via gradle -DZXING_CORE_ROOT=...
if(NOT DEFINED ZXING_CORE_ROOT)
    set(ZXING_CORE_ROOT
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../core"
        CACHE PATH "Path to zxing-cpp/core")
endif()

message(STATUS "[zxing-nanodet] ZXING_CORE_ROOT = ${ZXING_CORE_ROOT}")

# Normalize path separators (Windows backslash → forward slash) before glob
file(TO_CMAKE_PATH "${ZXING_CORE_ROOT}" ZXING_CORE_ROOT)
message(STATUS "[zxing-nanodet] ZXING_CORE_ROOT (normalized) = ${ZXING_CORE_ROOT}")

# Zint backend path — core/src/libzint/ files are Unix symlinks (broken on Windows git).
# Reference the actual source files from the zint/backend/ directory instead.
get_filename_component(ZINT_BACKEND_ROOT "${ZXING_CORE_ROOT}/../zint/backend" ABSOLUTE)
file(TO_CMAKE_PATH "${ZINT_BACKEND_ROOT}" ZINT_BACKEND_ROOT)
message(STATUS "[zxing-nanodet] ZINT_BACKEND_ROOT = ${ZINT_BACKEND_ROOT}")

# Generate Version.h from template (needed by several ZXing source files)
set(ZXING_READERS ON)
set(ZXING_WRITERS OFF)
set(PROJECT_VERSION_MAJOR 2)
set(PROJECT_VERSION_MINOR 3)
set(PROJECT_VERSION_PATCH 0)
configure_file("${ZXING_CORE_ROOT}/Version.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/Version.h")

# Collect ZXing core + NanoDet pre/post sources.
# Exclude NanoDetModelData.cpp (~5MB embedded bytes) — model is loaded from assets.
# Exclude test files.
file(GLOB_RECURSE ZXING_SOURCES "${ZXING_CORE_ROOT}/src/*.cpp")
list(FILTER ZXING_SOURCES EXCLUDE REGEX ".*NanoDetModelData\\.cpp$")
list(FILTER ZXING_SOURCES EXCLUDE REGEX ".*[Tt]est.*\\.cpp$")

add_library(zxing_nanodet_jni SHARED
    ZXingNanoDetJNI.cpp
    NanoDetModelDataStub.cpp                        # Stubs NANODET_MODEL_DATA/SIZE (model loaded from assets)
    "${ZXING_CORE_ROOT}/src/libzueci/zueci.c"       # zueci_* functions used by TextDecoder/TextEncoder
    ${ZXING_SOURCES}
)

target_include_directories(zxing_nanodet_jni PRIVATE
    "${ZXING_CORE_ROOT}/src"
    "${ZXING_CORE_ROOT}/src/onnx"
    "${ZXING_CORE_ROOT}/src/libzueci"   # zueci.h for zueci.c itself
    "${CMAKE_CURRENT_BINARY_DIR}"
)

target_compile_features(zxing_nanodet_jni PRIVATE cxx_std_17)

# Enable ZXing reader support (required for ReadBarcodes to actually decode)
target_compile_definitions(zxing_nanodet_jni PRIVATE ZXING_READERS)

# Enable all barcode format readers - these are checked by MultiFormatReader.cpp
# to instantiate the actual decoder objects (ZXING_WITH_1D, ZXING_WITH_QRCODE, etc.)
target_compile_definitions(zxing_nanodet_jni PRIVATE
    ZXING_WITH_1D
    ZXING_WITH_QRCODE
    ZXING_WITH_DATAMATRIX
    ZXING_WITH_AZTEC
    ZXING_WITH_PDF417
    ZXING_WITH_MAXICODE
)

# ZXING_USE_ONNXRUNTIME is deliberately NOT defined here.
# ORT inference runs in Kotlin via the Android ORT Java API.
# NanoDet.h exposes Preprocess/PostprocessGFL without requiring ORT C headers.

# 16 KB page-size alignment required for Android 15+ (API 35) compatibility.
# https://developer.android.com/guide/practices/page-sizes
target_link_options(zxing_nanodet_jni PRIVATE
    "-Wl,-z,max-page-size=16384"
)

target_link_libraries(zxing_nanodet_jni
    android
    log
)
