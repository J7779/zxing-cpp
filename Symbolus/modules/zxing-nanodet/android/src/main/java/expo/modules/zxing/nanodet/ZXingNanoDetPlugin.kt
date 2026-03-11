// SPDX-License-Identifier: Apache-2.0
// ZXingNanoDetPlugin.kt
//
// VisionCamera v4 frame processor plugin (Android).
// Plugin name: "detectBarcodes"
//
// Pipeline per frame (async):
//   callback(): YUV->RGBA copy (sync) -> dispatch to worker -> return cached results
//   worker:     preprocess -> ORT inference -> postprocess -> ZXing decode -> update cache

package expo.modules.zxing.nanodet

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.ImageFormat
import android.util.Log
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.VisionCameraProxy
import java.nio.FloatBuffer
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "ZXingNanoDetPlugin"
private const val MODEL_ASSET = "nanodet_barcode.onnx"

class ZXingNanoDetPlugin(
    proxy: VisionCameraProxy,
    options: Map<String, Any>?,
) : FrameProcessorPlugin() {

    private val appContext = proxy.context.applicationContext
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null

    // PP-OCRv5 fallback for barcodes ZXing cannot decode
    private val ocrEngine: OcrEngine by lazy { OcrEngine(appContext, ortEnv) }

    // Single background thread for NanoDet inference — avoids blocking the camera pipeline
    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "ZXingNanoDetWorker").apply { isDaemon = true }
    }
    // Thread pool for parallel ZXing/OCR decode (runs inside the NanoDet worker)
    private val decodePool = Executors.newFixedThreadPool(3) { r ->
        Thread(r, "DecodeWorker").apply { isDaemon = true }
    }
    private val isProcessing = AtomicBoolean(false)
    @Volatile private var cachedResults: List<Map<String, Any>> = emptyList()

    init {
        try {
            val modelBytes = appContext.assets.open(MODEL_ASSET).readBytes()
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            ortSession = ortEnv.createSession(modelBytes, sessionOptions)
            Log.i(TAG, "ORT session ready - inputs: ${ortSession?.inputNames}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to init ORT session: ${e.message}", e)
        }
    }

    // -- Frame processor callback (runs on VisionCamera thread) -------------------
    // Returns IMMEDIATELY with the last known results.
    // Copies RGBA bytes and dispatches heavy work to the background thread.

    override fun callback(frame: Frame, params: Map<String, Any>?): Any {
        val session = ortSession ?: return cachedResults

        val image = frame.image ?: return cachedResults
        val width  = image.width
        val height = image.height

        // If worker is still busy, skip this frame entirely
        if (!isProcessing.compareAndSet(false, true)) {
            return cachedResults
        }

        // Copy RGBA bytes synchronously - frame is only valid inside this callback
        val rgba = yuv420ToRGBA(image, width, height)
        image.close()

        if (rgba.isEmpty()) {
            isProcessing.set(false)
            return cachedResults
        }

        val confidence     = (params?.get("confidence")     as? Number)?.toFloat() ?: 0.35f
        val modelInputSize = (params?.get("modelInputSize") as? Number)?.toInt()   ?: 640
        val maxDetections  = (params?.get("maxDetections")  as? Number)?.toInt()   ?: 10
        val debug          = (params?.get("debug")          as? Boolean)           ?: false
        val enableZxing    = (params?.get("enableZxing")    as? Boolean)           ?: true
        val enableOcr      = (params?.get("enableOcr")      as? Boolean)           ?: true
        val enableDirectZxing     = (params?.get("enableDirectZxing")     as? Boolean) ?: false
        val zxingResolutionScale  = (params?.get("zxingResolutionScale")  as? Number)?.toFloat() ?: 1.0f
        val enableDamagedBarcode  = (params?.get("enableDamagedBarcode")  as? Boolean) ?: false
        // JS arrays arrive as List<*>; extract non-blank strings.
        @Suppress("UNCHECKED_CAST")
        val enabledFormats: Set<String>? = (params?.get("enabledFormats") as? List<*>)
            ?.mapNotNull { it as? String }
            ?.filter { it.isNotBlank() }
            ?.toHashSet()
            ?.takeIf { it.isNotEmpty() }

        // Dispatch all heavy work to background thread
        executor.execute {
            try {
                val results = runInference(
                    session, rgba, width, height,
                    confidence, modelInputSize, maxDetections, debug,
                    enableZxing, enableOcr, enabledFormats,
                    enableDirectZxing, zxingResolutionScale, enableDamagedBarcode
                )
                cachedResults = results
            } catch (e: Exception) {
                Log.e(TAG, "Inference error: ${e.message}", e)
            } finally {
                isProcessing.set(false)
            }
        }

        // Return last known results immediately (non-blocking)
        return cachedResults
    }

    // -- Background inference pipeline -------------------------------------------

    private fun runInference(
        session: OrtSession,
        rgba: ByteArray,
        width: Int,
        height: Int,
        confidence: Float,
        modelInputSize: Int,
        maxDetections: Int,
        debug: Boolean = false,
        enableZxing: Boolean = true,
        enableOcr: Boolean = true,
        enabledFormats: Set<String>? = null,   // null = accept all formats
        enableDirectZxing: Boolean = false,
        zxingResolutionScale: Float = 1.0f,
        enableDamagedBarcode: Boolean = false,
    ): List<Map<String, Any>> {
        val frameLog = mutableListOf<String>()
        fun log(msg: String) { Log.d(TAG, msg); if (debug) frameLog += msg }

        log("[FRAME] ${width}x${height} landscape=${width > height} debug=$debug enableZxing=$enableZxing enableOcr=$enableOcr enableDirectZxing=$enableDirectZxing zxingResScale=$zxingResolutionScale enableDamagedBarcode=$enableDamagedBarcode formats=${enabledFormats ?: "all"}")


        // 1. NanoDet preprocessing (C++)
        val preprocessed = ZXingNanoDetJNI.nativePreprocess(rgba, width, height, modelInputSize)
        val tensorSize = 3 * modelInputSize * modelInputSize
        if (preprocessed.size < tensorSize + 5) {
            log("[ERROR] preprocessed array too small: ${preprocessed.size}")
            return emptyList()
        }

        val scale = preprocessed[tensorSize]
        val padX  = preprocessed[tensorSize + 1]
        val padY  = preprocessed[tensorSize + 2]
        val newW  = preprocessed[tensorSize + 3].toInt()
        val newH  = preprocessed[tensorSize + 4].toInt()
        log("[NANODET_PRE] modelSize=$modelInputSize scale=$scale padX=$padX padY=$padY letterbox=${newW}x${newH}")

        // 2. ORT inference
        val inputShape  = longArrayOf(1, 3, modelInputSize.toLong(), modelInputSize.toLong())
        val tensorBuf   = FloatBuffer.wrap(preprocessed, 0, tensorSize)
        val inputTensor = OnnxTensor.createTensor(ortEnv, tensorBuf, inputShape)
        val inputName   = session.inputNames.first()
        log("[ORT] running session, input='$inputName'")
        val outputs     = session.run(mapOf(inputName to inputTensor))
        inputTensor.close()

        val outputTensor = outputs.get(session.outputNames.first()).get() as OnnxTensor
        val shape       = outputTensor.info.shape
        val outputBuf   = outputTensor.floatBuffer
        val outputArray = FloatArray(outputBuf.remaining()).also { outputBuf.get(it) }
        outputTensor.close()
        outputs.close()

        val numBoxes = if (shape.size >= 2) shape[1].toInt() else outputArray.size / 34
        val boxSize  = if (shape.size >= 3) shape[2].toInt() else 34
        log("[ORT_OUT] shape=${shape.toList()} numBoxes=$numBoxes boxSize=$boxSize")

        // 3. NanoDet postprocessing (C++)
        val boxes = ZXingNanoDetJNI.nativePostprocessGFL(
            outputArray, numBoxes, boxSize,
            width, height,
            scale, padX, padY,
            modelInputSize, confidence,
        )
        log("[NANODET_POST] detections=${boxes.size} (confidence threshold=$confidence)")
        for (i in boxes.indices) {
            val b = boxes[i]
            log("[DET#$i] x1=${b[0]} y1=${b[1]} x2=${b[2]} y2=${b[3]} score=${b[4]} class=${b[5]}")
        }

        // 4. ZXing + OCR decode per NanoDet-detected box (PARALLEL)
        val results = mutableListOf<Map<String, Any>>()
        val limit = minOf(boxes.size, maxDetections)

        // Prepare crop data for all detections
        data class CropInfo(
            val idx: Int,
            val bx1: Int, val by1: Int, val bx2: Int, val by2: Int,
            val cx: Int, val cy: Int, val cw: Int, val ch: Int,
            val detScore: Float,
            val debugCropBase64: String?,
        )

        val crops = (0 until limit).mapNotNull { i ->
            val b = boxes[i]
            val bx1i = b[0].toFloat().toInt()
            val by1i = b[1].toFloat().toInt()
            val bx2i = b[2].toFloat().toInt()
            val by2i = b[3].toFloat().toInt()
            val detScore = b[4].toFloat()

            val padW = ((bx2i - bx1i) * 0.3f).toInt()
            val padH = ((by2i - by1i) * 0.3f).toInt()
            val cxi = maxOf(0, bx1i - padW)
            val cyi = maxOf(0, by1i - padH)
            val cwi = minOf(width  - cxi, bx2i - bx1i + 2 * padW)
            val chi = minOf(height - cyi, by2i - by1i + 2 * padH)
            log("[CROP#$i] raw=($bx1i,$by1i)-($bx2i,$by2i) padded=($cxi,$cyi ${cwi}x${chi}) score=$detScore")
            if (cwi <= 0 || chi <= 0) { log("[CROP#$i] SKIPPED (zero area after padding)"); return@mapNotNull null }

            val debugCropBase64: String? = if (debug) generateZxingInputBase64(rgba, width, height, cxi, cyi, cwi, chi) else null
            CropInfo(i, bx1i, by1i, bx2i, by2i, cxi, cyi, cwi, chi, detScore, debugCropBase64)
        }

        // Submit ZXing decode per crop to the thread pool
        val decodeFutures: List<Pair<CropInfo, Future<List<Map<String, Any>>>>> = crops.map { crop ->
            crop to decodePool.submit(Callable {
                decodeSingleCrop(
                    crop.idx, rgba, width, height,
                    crop.bx1, crop.by1, crop.bx2, crop.by2,
                    crop.cx, crop.cy, crop.cw, crop.ch,
                    crop.detScore, crop.debugCropBase64,
                    debug, enableZxing, enableOcr, enabledFormats,
                    enableDamagedBarcode,
                )
            })
        }

        // Submit direct ZXing concurrently with per-crop decoding
        val directFuture: Future<List<Map<String, Any>>>? =
            if (enableDirectZxing && enableZxing) {
                decodePool.submit(Callable {
                    decodeDirectZxing(rgba, width, height, debug, enabledFormats)
                })
            } else null

        // Collect per-crop results (preserves detection order)
        for ((_, future) in decodeFutures) {
            try { results += future.get() } catch (e: Exception) {
                Log.e(TAG, "Decode error: ${e.message}", e)
            }
        }

        // Collect direct ZXing results, de-duplicating
        if (directFuture != null) {
            try {
                val directResults = directFuture.get()
                val existingTexts = results.mapNotNull { it["text"] as? String }.toHashSet()
                for (r in directResults) {
                    val text = r["text"] as? String ?: ""
                    if (text.isNotBlank() && !existingTexts.contains(text)) {
                        results += r
                        existingTexts += text
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Direct ZXing error: ${e.message}", e)
            }
        }

        if (debug && results.isEmpty() && frameLog.isNotEmpty()) {
            // No detections at all — still surface logs via a sentinel entry
            results += mutableMapOf<String, Any>(
                "format" to "__debug__", "text" to "",
                "confidence" to 0.0,
                "boundingBox" to mapOf("x" to 0.0, "y" to 0.0, "width" to 0.0, "height" to 0.0),
                "cornerPoints" to emptyList<Any>(),
                "debugLogs" to frameLog.toList(),
            )
        }

        return results
    }

    // -- Helper: build an UNKNOWN result entry ------------------------------------

    private fun buildUnknownResult(
        bx1: Int, by1: Int, bx2: Int, by2: Int, score: Float,
        debug: Boolean, debugCropBase64: String?, snapshotLogs: List<String>?,
    ): Map<String, Any> {
        val resultMap = mutableMapOf<String, Any>(
            "format"         to "UNKNOWN",
            "text"           to "",
            "confidence"     to score.toDouble(),
            "isOcrFallback"  to false,
            "source"         to "nanodet",
            "boundingBox"    to mapOf(
                "x"      to bx1.toDouble(),
                "y"      to by1.toDouble(),
                "width"  to (bx2 - bx1).toDouble(),
                "height" to (by2 - by1).toDouble(),
            ),
            "cornerPoints" to emptyList<Any>(),
        )
        if (debug) {
            if (debugCropBase64 != null) resultMap["debugCropBase64"] = debugCropBase64
            if (snapshotLogs != null) resultMap["debugLogs"] = snapshotLogs
        }
        return resultMap
    }

    // -- Decode a single NanoDet crop (ZXing + OCR fallback) ─────────────────
    // Designed to be called from the decode thread pool.

    private fun decodeSingleCrop(
        idx: Int,
        rgba: ByteArray, width: Int, height: Int,
        bx1: Int, by1: Int, bx2: Int, by2: Int,
        cx: Int, cy: Int, cw: Int, ch: Int,
        detScore: Float, debugCropBase64: String?,
        debug: Boolean,
        enableZxing: Boolean, enableOcr: Boolean,
        enabledFormats: Set<String>?,
        enableDamagedBarcode: Boolean,
    ): List<Map<String, Any>> {
        val results = mutableListOf<Map<String, Any>>()

        // ZXing decode
        val rawBarcodes = if (enableZxing) {
            ZXingNanoDetJNI.nativeDecodeBarcode(rgba, width, height, cx, cy, cw, ch, debug, enableDamagedBarcode)
        } else {
            emptyArray()
        }

        // Extract JNI log entry
        var startIdx = 0
        if (debug && rawBarcodes.isNotEmpty() && rawBarcodes[0].getOrNull(0) == "__log__") {
            startIdx = 1
        }
        val allBarcodes = if (startIdx > 0) rawBarcodes.drop(startIdx) else rawBarcodes.toList()

        // Format filter
        val barcodes = if (enabledFormats != null) {
            allBarcodes.filter { bc -> enabledFormats.contains(bc.getOrNull(0) ?: "") }
        } else {
            allBarcodes
        }

        val snapshotLogs: List<String>? = null  // debug logs not thread-safe; skip in worker

        if (barcodes.isNotEmpty()) {
            for (barcode in barcodes) {
                val cornerPoints = (0..3).map { c ->
                    mapOf(
                        "x" to barcode[2 + c * 2].toDouble(),
                        "y" to barcode[3 + c * 2].toDouble(),
                    )
                }
                val resultMap = mutableMapOf<String, Any>(
                    "format"      to barcode[0],
                    "text"        to barcode[1],
                    "confidence"  to detScore.toDouble(),
                    "source"      to "nanodet",
                    "boundingBox" to mapOf(
                        "x"      to cx.toDouble(),
                        "y"      to cy.toDouble(),
                        "width"  to cw.toDouble(),
                        "height" to ch.toDouble(),
                    ),
                    "cornerPoints" to cornerPoints,
                )
                if (debug && debugCropBase64 != null) resultMap["debugCropBase64"] = debugCropBase64
                results += resultMap
            }
        } else if (enableDamagedBarcode && enableZxing && enableOcr && ocrEngine.isAvailable) {
            // Damaged barcode merge: ZXing partial + OCR
            val partialZxing = allBarcodes.firstOrNull()?.getOrNull(1) ?: ""
            val ocrText = ocrEngine.recognizeTextInRegion(rgba, width, height, cx, cy, cw, ch)
            val mergedText = mergePartialBarcodeTexts(partialZxing, ocrText)

            if (mergedText.isNotBlank()) {
                val resultMap = mutableMapOf<String, Any>(
                    "format"         to (allBarcodes.firstOrNull()?.getOrNull(0) ?: "OCR"),
                    "text"           to mergedText,
                    "confidence"     to detScore.toDouble(),
                    "isOcrFallback"  to true,
                    "source"         to "nanodet",
                    "mergedText"     to mergedText,
                    "boundingBox"    to mapOf(
                        "x"      to bx1.toDouble(),
                        "y"      to by1.toDouble(),
                        "width"  to (bx2 - bx1).toDouble(),
                        "height" to (by2 - by1).toDouble(),
                    ),
                    "cornerPoints" to emptyList<Any>(),
                )
                if (debug && debugCropBase64 != null) resultMap["debugCropBase64"] = debugCropBase64
                results += resultMap
            } else {
                results += buildUnknownResult(bx1, by1, bx2, by2, detScore, debug, debugCropBase64, snapshotLogs)
            }
        } else {
            // OCR fallback only
            val ocrText = if (enableOcr && ocrEngine.isAvailable) {
                ocrEngine.recognizeTextInRegion(rgba, width, height, cx, cy, cw, ch)
            } else ""

            val resultMap = mutableMapOf<String, Any>(
                "format"         to if (ocrText.isNotBlank()) "OCR" else "UNKNOWN",
                "text"           to ocrText,
                "confidence"     to detScore.toDouble(),
                "isOcrFallback"  to ocrText.isNotBlank(),
                "source"         to "nanodet",
                "boundingBox"    to mapOf(
                    "x"      to bx1.toDouble(),
                    "y"      to by1.toDouble(),
                    "width"  to (bx2 - bx1).toDouble(),
                    "height" to (by2 - by1).toDouble(),
                ),
                "cornerPoints" to emptyList<Any>(),
            )
            if (debug && debugCropBase64 != null) resultMap["debugCropBase64"] = debugCropBase64
            results += resultMap
        }

        return results
    }

    // -- Decode full-frame ZXing (direct pass) ──────────────────────────────
    // Designed to be called from the decode thread pool.

    private fun decodeDirectZxing(
        rgba: ByteArray, width: Int, height: Int,
        debug: Boolean, enabledFormats: Set<String>?,
    ): List<Map<String, Any>> {
        val results = mutableListOf<Map<String, Any>>()

        val directBarcodes = ZXingNanoDetJNI.nativeDecodeBarcode(
            rgba, width, height, 0, 0, width, height, debug, false
        )
        var dStartIdx = 0
        if (debug && directBarcodes.isNotEmpty() && directBarcodes[0].getOrNull(0) == "__log__") {
            dStartIdx = 1
        }
        val dBarcodes = if (dStartIdx > 0) directBarcodes.drop(dStartIdx) else directBarcodes.toList()

        val filtered = if (enabledFormats != null) {
            dBarcodes.filter { bc -> enabledFormats.contains(bc.getOrNull(0) ?: "") }
        } else {
            dBarcodes
        }

        for (barcode in filtered) {
            val text = barcode.getOrNull(1) ?: ""
            if (text.isBlank()) continue
            val cornerPoints = (0..3).map { c ->
                mapOf(
                    "x" to barcode[2 + c * 2].toDouble(),
                    "y" to barcode[3 + c * 2].toDouble(),
                )
            }
            results += mutableMapOf<String, Any>(
                "format"      to barcode[0],
                "text"        to text,
                "confidence"  to 0.5,
                "source"      to "direct",
                "boundingBox" to mapOf(
                    "x"      to 0.0,
                    "y"      to 0.0,
                    "width"  to width.toDouble(),
                    "height" to height.toDouble(),
                ),
                "cornerPoints" to cornerPoints,
            )
        }

        return results
    }

    // -- Helper: merge partial barcode readings from ZXing and OCR ---------------
    // For damaged barcodes: ZXing might decode some digits, OCR reads the rest.
    // We align overlapping characters and fill gaps from the longer source.

    private fun mergePartialBarcodeTexts(zxingText: String, ocrText: String): String {
        if (zxingText.isBlank()) return ocrText.trim()
        if (ocrText.isBlank()) return zxingText.trim()

        // Strip non-alphanumeric from OCR to normalise noise
        val cleanOcr = ocrText.filter { it.isLetterOrDigit() || it == '-' }
        val cleanZxing = zxingText.trim()

        // If one contains the other, pick the longer one
        if (cleanZxing.contains(cleanOcr)) return cleanZxing
        if (cleanOcr.contains(cleanZxing)) return cleanOcr

        // Try to find overlap: end of one matches start of the other
        val merged = findOverlapMerge(cleanZxing, cleanOcr)
            ?: findOverlapMerge(cleanOcr, cleanZxing)
        if (merged != null) return merged

        // If zxing decoded more characters, prefer it; otherwise concatenate
        return if (cleanZxing.length >= cleanOcr.length) cleanZxing else cleanOcr
    }

    private fun findOverlapMerge(a: String, b: String): String? {
        // Find the longest suffix of a that matches a prefix of b
        val maxOverlap = minOf(a.length, b.length)
        for (len in maxOverlap downTo 1) {
            if (a.endsWith(b.substring(0, len))) {
                return a + b.substring(len)
            }
        }
        return null
    }

    // -- Debug: encode the EXACT image that ZXing decodes (post-rotation) ------
    // Replicates the same 90° CW rotation the JNI layer applies when the sensor
    // frame is landscape, so the thumbnail matches what ZXing actually sees.

    private fun generateZxingInputBase64(
        rgba: ByteArray,
        frameWidth: Int,
        frameHeight: Int,
        cx: Int, cy: Int, cw: Int, ch: Int,
    ): String? = try {
        val pixels = IntArray(cw * ch) { idx ->
            val row = idx / cw
            val col = idx % cw
            val srcIdx = ((cy + row) * frameWidth + (cx + col)) * 4
            val r = rgba[srcIdx].toInt() and 0xFF
            val g = rgba[srcIdx + 1].toInt() and 0xFF
            val b = rgba[srcIdx + 2].toInt() and 0xFF
            val gray = (r * 77 + g * 150 + b * 29) shr 8
            android.graphics.Color.rgb(gray, gray, gray)
        }
        var bmp = android.graphics.Bitmap.createBitmap(pixels, cw, ch, android.graphics.Bitmap.Config.ARGB_8888)
        // Apply the same 90° CW rotation as the JNI ZXing path
        if (frameWidth > frameHeight) {
            val matrix = android.graphics.Matrix().apply { postRotate(90f) }
            val rotated = android.graphics.Bitmap.createBitmap(bmp, 0, 0, cw, ch, matrix, true)
            bmp.recycle()
            bmp = rotated
        }
        val baos = java.io.ByteArrayOutputStream()
        bmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, 80, baos)
        bmp.recycle()
        android.util.Base64.encodeToString(baos.toByteArray(), android.util.Base64.NO_WRAP)
    } catch (e: Exception) {
        Log.e(TAG, "generateZxingInputBase64 error: ${e.message}")
        null
    }

    // -- YUV_420_888 -> RGBA conversion -----------------------------------------

    private fun yuv420ToRGBA(image: android.media.Image, width: Int, height: Int): ByteArray {
        if (image.format != ImageFormat.YUV_420_888) return ByteArray(0)

        val planes        = image.planes
        val yBuf          = planes[0].buffer
        val uBuf          = planes[1].buffer
        val vBuf          = planes[2].buffer
        val yRowStride    = planes[0].rowStride
        val uvRowStride   = planes[1].rowStride
        val uvPixelStride = planes[1].pixelStride

        val rgba   = ByteArray(width * height * 4)
        var outIdx = 0

        for (row in 0 until height) {
            for (col in 0 until width) {
                val y = yBuf[row * yRowStride + col].toInt() and 0xFF
                val uvRow = row / 2
                val uvCol = col / 2
                val uvIdx = uvRow * uvRowStride + uvCol * uvPixelStride
                val u = (uBuf[uvIdx].toInt() and 0xFF) - 128
                val v = (vBuf[uvIdx].toInt() and 0xFF) - 128

                val r = clamp(y + (1.370705f * v).toInt())
                val g = clamp(y - (0.337633f * u).toInt() - (0.698001f * v).toInt())
                val b = clamp(y + (1.732446f * u).toInt())

                rgba[outIdx++] = r.toByte()
                rgba[outIdx++] = g.toByte()
                rgba[outIdx++] = b.toByte()
                rgba[outIdx++] = 255.toByte()
            }
        }
        return rgba
    }

    private fun clamp(v: Int): Int = v.coerceIn(0, 255)

    // -- Plugin registration ----------------------------------------------------

    companion object {
        fun register(proxy: VisionCameraProxy, options: Map<String, Any>?): ZXingNanoDetPlugin =
            ZXingNanoDetPlugin(proxy, options)
    }
}