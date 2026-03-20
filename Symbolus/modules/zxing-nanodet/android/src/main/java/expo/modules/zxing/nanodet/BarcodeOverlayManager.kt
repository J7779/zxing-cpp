// SPDX-License-Identifier: Apache-2.0
// Thread-safe singleton holding the latest ZXing barcode detections.
// The FrameProcessorPlugin writes here; BarcodeOverlayView reads and draws.

package expo.modules.zxing.nanodet

import android.os.Handler
import android.os.Looper

/**
 * One detected barcode with ZXing-returned position data.
 */
data class OverlayBarcode(
    val format: String,
    val text: String,
    /** 4 corner points from ZXing (frame pixel coordinates). */
    val corners: List<Pair<Float, Float>>,
    /** Bounding box in frame coordinates: x, y, width, height. */
    val bboxX: Float,
    val bboxY: Float,
    val bboxW: Float,
    val bboxH: Float,
)

/**
 * Thread-safe singleton that bridges detection results from the background
 * decode thread to the UI overlay view.
 */
object BarcodeOverlayManager {
    private val handler = Handler(Looper.getMainLooper())

    @Volatile var frameWidth: Int = 0
        private set
    @Volatile var frameHeight: Int = 0
        private set
    @Volatile var barcodes: List<OverlayBarcode> = emptyList()
        private set

    private val listeners = mutableSetOf<() -> Unit>()

    fun addListener(listener: () -> Unit) {
        synchronized(listeners) { listeners.add(listener) }
    }

    fun removeListener(listener: () -> Unit) {
        synchronized(listeners) { listeners.remove(listener) }
    }

    /**
     * Called from the decode worker thread after each inference.
     * Posts an invalidation to the UI thread.
     */
    fun update(fw: Int, fh: Int, results: List<OverlayBarcode>) {
        frameWidth = fw
        frameHeight = fh
        barcodes = results
        handler.post {
            synchronized(listeners) {
                for (l in listeners) l()
            }
        }
    }
}
