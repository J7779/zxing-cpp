// SPDX-License-Identifier: Apache-2.0
// VisionCamera plugin registry entry point for Android.
// Registers "detectBarcodes" so VisionCamera's JS-side
// createFrameProcessorPlugin("detectBarcodes") can resolve this class.

package expo.modules.zxing.nanodet

import com.mrousavy.camera.frameprocessors.FrameProcessorPluginRegistry

class ZXingNanoDetPluginPackage {
    companion object {
        @JvmStatic
        fun init() {
            FrameProcessorPluginRegistry.addFrameProcessorPlugin("detectBarcodes") { proxy, options ->
                ZXingNanoDetPlugin(proxy, options)
            }
        }
    }
}
