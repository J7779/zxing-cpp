// SPDX-License-Identifier: Apache-2.0
// NanoDetModelDataStub.cpp
//
// Stub definitions for NANODET_MODEL_DATA and NANODET_MODEL_SIZE.
// The Android build loads the ONNX model from assets via Kotlin (ORT Java API),
// so the compiled-in model bytes are not needed. These stubs satisfy the linker
// since NanoDet.cpp unconditionally defines GetModelData()/GetModelSize() which
// reference these symbols, even though those functions are never called here.

// 'extern' is required in C++ — plain 'const' at namespace scope has internal linkage.
extern const unsigned char NANODET_MODEL_DATA[] = {0};
extern const unsigned int  NANODET_MODEL_SIZE   = 0;
