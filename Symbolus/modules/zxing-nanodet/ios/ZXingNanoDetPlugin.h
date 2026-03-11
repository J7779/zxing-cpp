// SPDX-License-Identifier: Apache-2.0
// ZXingNanoDetPlugin.h — Objective-C++ header for the VisionCamera frame processor plugin.

#pragma once

#import <Foundation/Foundation.h>
#import <VisionCamera/FrameProcessorPlugin.h>
#import <VisionCamera/FrameProcessorPluginRegistry.h>

NS_ASSUME_NONNULL_BEGIN

/// VisionCamera v4 frame processor plugin that runs the NanoDet barcode
/// detection model followed by ZXing decoding on every camera frame.
@interface ZXingNanoDetPlugin : FrameProcessorPlugin

@end

NS_ASSUME_NONNULL_END
