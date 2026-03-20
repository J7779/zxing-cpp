// SPDX-License-Identifier: Apache-2.0
// React Native ViewManager that exposes BarcodeOverlayView as a native component.
// Usage from JS/TS:
//   import { requireNativeComponent } from 'react-native';
//   const NativeBarcodeOverlay = requireNativeComponent('BarcodeOverlayView');
//   <NativeBarcodeOverlay style={StyleSheet.absoluteFill} visible={true} />

package expo.modules.zxing.nanodet

import com.facebook.react.uimanager.SimpleViewManager
import com.facebook.react.uimanager.ThemedReactContext
import com.facebook.react.uimanager.annotations.ReactProp

class BarcodeOverlayViewManager : SimpleViewManager<BarcodeOverlayView>() {

    override fun getName(): String = "BarcodeOverlayView"

    override fun createViewInstance(reactContext: ThemedReactContext): BarcodeOverlayView {
        return BarcodeOverlayView(reactContext)
    }

    @ReactProp(name = "visible", defaultBoolean = true)
    fun setVisible(view: BarcodeOverlayView, visible: Boolean) {
        view.overlayVisible = visible
    }

    @ReactProp(name = "mirrorX", defaultBoolean = false)
    fun setMirrorX(view: BarcodeOverlayView, mirror: Boolean) {
        view.mirrorX = mirror
    }
}
