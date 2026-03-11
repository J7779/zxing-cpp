// SPDX-License-Identifier: Apache-2.0
// Kotlin JNI stubs matching ZXingNanoDetJNI.cpp

package expo.modules.zxing.nanodet

object ZXingNanoDetJNI {
    init {
        System.loadLibrary("zxing_nanodet_jni")
    }

    /**
     * Runs NanoDet letterbox + BGR normalization on an RGBA frame.
     *
     * Returns a FloatArray of length (3 * targetSize * targetSize + 5):
     *   [0 .. N-1]  = CHW BGR normalized tensor
     *   [N]         = scale factor
     *   [N+1]       = padX (as float)
     *   [N+2]       = padY (as float)
     *   [N+3]       = newWidth (as float)
     *   [N+4]       = newHeight (as float)
     */
    external fun nativePreprocess(
        rgba: ByteArray,
        width: Int,
        height: Int,
        targetSize: Int,
    ): FloatArray

    /**
     * Decodes NanoDet GFL output tensor and applies NMS.
     *
     * @param output     Flat float array from ORT session output [numBoxes * boxSize]
     * @param numBoxes   Number of anchor boxes (e.g. 3598 for 416×416)
     * @param boxSize    Values per box (e.g. 34 = 2 classes + 32 DFL)
     * @return Array of String[6]: [x1, y1, x2, y2, score, classId]
     */
    external fun nativePostprocessGFL(
        output: FloatArray,
        numBoxes: Int,
        boxSize: Int,
        srcW: Int,
        srcH: Int,
        scale: Float,
        padX: Float,
        padY: Float,
        targetSize: Int,
        confidence: Float,
    ): Array<Array<String>>

    /**
     * Crops the RGBA frame at [cropX, cropY, cropW, cropH], converts to luma,
     * rotates 90° CW when frameW > frameH, then runs ZXing ReadBarcodes.
     *
     * When debug=true, the FIRST element of the returned array is a log entry:
     *   result[0] = ["__log__", "line1\nline2\n..."]
     * Subsequent elements are normal decode results:
     *   [format, text, cx0, cy0, cx1, cy1, cx2, cy2, cx3, cy3]
     */
    external fun nativeDecodeBarcode(
        rgba: ByteArray,
        frameW: Int,
        frameH: Int,
        cropX: Int,
        cropY: Int,
        cropW: Int,
        cropH: Int,
        debug: Boolean,
        enableDamagedBarcode: Boolean = false,
    ): Array<Array<String>>
}

