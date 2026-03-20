// SPDX-License-Identifier: Apache-2.0
// React Native native module package registration for Camera2ISPModule.
// This is NOT an Expo module — it's a standard RN native module registered
// alongside the VisionCamera frame processor plugin.

package expo.modules.zxing.nanodet

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

@Suppress("OVERRIDE_DEPRECATION")
class Camera2ISPPackage : ReactPackage {
    override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> {
        return listOf(Camera2ISPModule(reactContext))
    }

    override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> {
        return listOf(BarcodeOverlayViewManager())
    }
}
