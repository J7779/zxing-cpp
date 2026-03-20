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
import java.util.concurrent.atomic.AtomicLong

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
    // Thread pool for parallel ZXing decode (runs inside the NanoDet worker)
    private val decodePool = Executors.newFixedThreadPool(3) { r ->
        Thread(r, "DecodeWorker").apply { isDaemon = true }
    }
    // Dedicated single thread for OCR — never blocks the decode pipeline
    private val ocrExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "OcrWorker").apply { isDaemon = true }
    }
    private val isProcessing = AtomicBoolean(false)
    @Volatile private var cachedResults: List<Map<String, Any>> = emptyList()
    // Async OCR result from previous frame — injected into next inference results
    @Volatile private var pendingOcrResult: Map<String, Any>? = null
    // Monotonically increasing counter — tags each inference so JS can detect new results
    private val inferenceCounter = AtomicLong(0)

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
        val enableOcr      = (params?.get("enableOcr")      as? Boolean)           ?: false
        val enableNanoDet  = (params?.get("enableNanoDet")  as? Boolean)           ?: false
        val enableDirectZxing     = (params?.get("enableDirectZxing")     as? Boolean) ?: true
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
                    enableDirectZxing, zxingResolutionScale, enableDamagedBarcode,
                    enableNanoDet
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
        enableDirectZxing: Boolean = true,
        zxingResolutionScale: Float = 1.0f,
        enableDamagedBarcode: Boolean = false,
        enableNanoDet: Boolean = false,
    ): List<Map<String, Any>> {
        val frameLog = mutableListOf<String>()
        fun log(msg: String) { Log.d(TAG, msg); if (debug) frameLog += msg }

        log("[FRAME] ${width}x${height} landscape=${width > height} debug=$debug enableZxing=$enableZxing enableOcr=$enableOcr enableNanoDet=$enableNanoDet enableDirectZxing=$enableDirectZxing zxingResScale=$zxingResolutionScale enableDamagedBarcode=$enableDamagedBarcode formats=${enabledFormats ?: "all"}")
        val pipelineStartMs = System.currentTimeMillis()

        val results = mutableListOf<Map<String, Any>>()

        // ── NanoDet detection pipeline (skipped when enableNanoDet=false) ────────
        if (enableNanoDet) {

        // 1. NanoDet preprocessing (C++)
        val preStartMs = System.currentTimeMillis()
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
        log("[NANODET_PRE] modelSize=$modelInputSize scale=$scale padX=$padX padY=$padY letterbox=${newW}x${newH} preprocess_ms=${System.currentTimeMillis() - preStartMs}")

        // 2. ORT inference
        val ortStartMs = System.currentTimeMillis()
        val inputShape  = longArrayOf(1, 3, modelInputSize.toLong(), modelInputSize.toLong())
        val tensorBuf   = FloatBuffer.wrap(preprocessed, 0, tensorSize)
        val inputTensor = OnnxTensor.createTensor(ortEnv, tensorBuf, inputShape)
        val inputName   = session.inputNames.first()
        log("[ORT] running session, input='$inputName'")
        val outputs     = session.run(mapOf(inputName to inputTensor))
        val ortDurationMs = System.currentTimeMillis() - ortStartMs
        inputTensor.close()

        val outputTensor = outputs.get(session.outputNames.first()).get() as OnnxTensor
        val shape       = outputTensor.info.shape
        val outputBuf   = outputTensor.floatBuffer
        val outputArray = FloatArray(outputBuf.remaining()).also { outputBuf.get(it) }
        outputTensor.close()
        outputs.close()

        val numBoxes = if (shape.size >= 2) shape[1].toInt() else outputArray.size / 34
        val boxSize  = if (shape.size >= 3) shape[2].toInt() else 34
        log("[ORT_OUT] shape=${shape.toList()} numBoxes=$numBoxes boxSize=$boxSize ort_ms=$ortDurationMs")

        // 3. NanoDet postprocessing (C++)
        val postStartMs = System.currentTimeMillis()
        val boxes = ZXingNanoDetJNI.nativePostprocessGFL(
            outputArray, numBoxes, boxSize,
            width, height,
            scale, padX, padY,
            modelInputSize, confidence,
        )
        log("[NANODET_POST] detections=${boxes.size} (confidence threshold=$confidence) postprocess_ms=${System.currentTimeMillis() - postStartMs}")
        for (i in boxes.indices) {
            val b = boxes[i]
            log("[DET#$i] x1=${b[0]} y1=${b[1]} x2=${b[2]} y2=${b[3]} score=${b[4]} class=${b[5]}")
        }

        // 4. ZXing + OCR decode per NanoDet-detected box (PARALLEL)
        val limit = minOf(boxes.size, maxDetections)
        // Only one crop per frame may trigger async OCR (avoid queueing N slow inferences)
        val ocrSlotTaken = AtomicBoolean(false)

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
                    enableDamagedBarcode, ocrSlotTaken,
                )
            })
        }

        // Collect per-crop results (preserves detection order)
        for ((_, future) in decodeFutures) {
            try { results += future.get() } catch (e: Exception) {
                Log.e(TAG, "Decode error: ${e.message}", e)
            }
        }

        } else {
            log("[NANODET_SKIP] NanoDet disabled, using direct ZXing only")
        } // end if (enableNanoDet)

        // ── Direct full-frame ZXing (always runs when enabled, primary path when NanoDet off) ──
        if (enableDirectZxing && enableZxing) {
            try {
                val directResults = decodeDirectZxing(rgba, width, height, debug, enabledFormats)
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

        // Always attach pipeline-level frameLog to every result when debug is on
        if (debug && frameLog.isNotEmpty()) {
            for (result in results) {
                val m = result as? MutableMap<String, Any> ?: continue
                @Suppress("UNCHECKED_CAST")
                val existing = m["debugLogs"] as? List<String> ?: emptyList()
                m["debugLogs"] = frameLog + existing
            }
        }

        val totalPipelineMs = System.currentTimeMillis() - pipelineStartMs
        log("[PIPELINE_SUMMARY] total_ms=$totalPipelineMs results=${results.size} validBarcodes=${results.count { (it["format"] as? String ?: "") != "UNKNOWN" && (it["format"] as? String ?: "") != "__debug__" }}")

        // Inject any pending async OCR result from the previous frame's background run
        pendingOcrResult?.let { ocrResult ->
            val existingTexts = results.mapNotNull { it["text"] as? String }.toHashSet()
            val ocrText = ocrResult["text"] as? String ?: ""
            if (ocrText.isNotBlank() && !existingTexts.contains(ocrText)) {
                results += ocrResult
            }
            pendingOcrResult = null
        }

        // Tag every result with a unique inference ID so the JS consensus algorithm
        // can distinguish genuinely new inference frames from cached repeats.
        // Convert to Double — VisionCamera's JSI bridge cannot serialise java.lang.Long.
        val infId = inferenceCounter.incrementAndGet().toDouble()
        for (result in results) {
            (result as? MutableMap<String, Any>)?.set("_inferenceId", infId)
        }

        // Push ZXing bounding box / corner data to the native overlay view
        updateOverlay(width, height, results)

        return results
    }

    /** Convert result maps to OverlayBarcode list and push to the singleton. */
    @Suppress("UNCHECKED_CAST")
    private fun updateOverlay(frameW: Int, frameH: Int, results: List<Map<String, Any>>) {
        val overlayBarcodes = results.mapNotNull { r ->
            val format = r["format"] as? String ?: return@mapNotNull null
            if (format == "UNKNOWN" || format == "__debug__") return@mapNotNull null
            val text = r["text"] as? String ?: ""
            val corners = (r["cornerPoints"] as? List<Map<String, Any>>)?.map { pt ->
                val x = (pt["x"] as? Number)?.toFloat() ?: 0f
                val y = (pt["y"] as? Number)?.toFloat() ?: 0f
                x to y
            } ?: emptyList()
            val bbox = r["boundingBox"] as? Map<String, Any>
            val bx = (bbox?.get("x") as? Number)?.toFloat() ?: 0f
            val by = (bbox?.get("y") as? Number)?.toFloat() ?: 0f
            val bw = (bbox?.get("width") as? Number)?.toFloat() ?: 0f
            val bh = (bbox?.get("height") as? Number)?.toFloat() ?: 0f
            OverlayBarcode(format, text, corners, bx, by, bw, bh)
        }
        BarcodeOverlayManager.update(frameW, frameH, overlayBarcodes)
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
        ocrSlotTaken: AtomicBoolean,
    ): List<Map<String, Any>> {
        val results = mutableListOf<Map<String, Any>>()
        val cropLog = mutableListOf<String>()
        fun clog(msg: String) { Log.d(TAG, "[CROP#$idx] $msg"); if (debug) cropLog += "[CROP#$idx] $msg" }

        clog("START box=($bx1,$by1)-($bx2,$by2) padded=($cx,$cy ${cw}x${ch}) score=$detScore enableZxing=$enableZxing enableDamagedBarcode=$enableDamagedBarcode")

        // ZXing decode
        val zxingStartMs = System.currentTimeMillis()
        val rawBarcodes = if (enableZxing) {
            ZXingNanoDetJNI.nativeDecodeBarcode(rgba, width, height, cx, cy, cw, ch, debug, enableDamagedBarcode)
        } else {
            emptyArray()
        }
        val zxingDurationMs = System.currentTimeMillis() - zxingStartMs
        clog("ZXing returned ${rawBarcodes.size} entries in ${zxingDurationMs}ms")

        // Extract JNI log entry
        var startIdx = 0
        val jniLogLines = mutableListOf<String>()
        if (debug && rawBarcodes.isNotEmpty() && rawBarcodes[0].getOrNull(0) == "__log__") {
            startIdx = 1
            val logText = rawBarcodes[0].getOrNull(1) ?: ""
            jniLogLines.addAll(logText.split("\n").filter { it.isNotBlank() })
            clog("JNI produced ${jniLogLines.size} diagnostic lines")
        }
        val allBarcodes = if (startIdx > 0) rawBarcodes.drop(startIdx) else rawBarcodes.toList()
        clog("allBarcodes=${allBarcodes.size} (after removing log entry)")

        // Format filter
        val barcodes = if (enabledFormats != null) {
            allBarcodes.filter { bc -> enabledFormats.contains(bc.getOrNull(0) ?: "") }
        } else {
            allBarcodes
        }
        clog("filteredBarcodes=${barcodes.size} enabledFormats=${enabledFormats ?: "all"}")
        for (i in barcodes.indices) {
            clog("barcode[$i] format=${barcodes[i].getOrNull(0)} text=${barcodes[i].getOrNull(1)?.take(40)}")
        }

        // Combine all debug logs: crop-level + JNI C++ logs
        val allDebugLogs = if (debug) (cropLog + jniLogLines) else null

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
                if (debug && allDebugLogs != null) resultMap["debugLogs"] = allDebugLogs
                results += resultMap
            }
        } else if (enableDamagedBarcode && enableZxing && enableOcr && ocrEngine.isAvailable) {
            // Damaged barcode merge: ZXing partial + OCR (async to avoid blocking)
            val partialZxing = allBarcodes.firstOrNull()?.getOrNull(1) ?: ""

            if (ocrSlotTaken.compareAndSet(false, true)) {
                // Submit OCR to dedicated thread — result injected on next frame
                val rgbaCopy = rgba // already a ByteArray copy; safe to close over
                ocrExecutor.submit {
                    try {
                        val ocrText = ocrEngine.recognizeTextInRegion(rgbaCopy, width, height, cx, cy, cw, ch)
                        val mergedText = mergePartialBarcodeTexts(partialZxing, ocrText)
                        if (mergedText.isNotBlank()) {
                            pendingOcrResult = mutableMapOf<String, Any>(
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
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Async damaged OCR error: ${e.message}", e)
                    }
                }
            }
            // Return UNKNOWN immediately — OCR result arrives next frame
            results += buildUnknownResult(bx1, by1, bx2, by2, detScore, debug, debugCropBase64, allDebugLogs)
        } else {
            // OCR fallback (async — submit to dedicated thread, return UNKNOWN now)
            if (enableOcr && ocrEngine.isAvailable && ocrSlotTaken.compareAndSet(false, true)) {
                val rgbaCopy = rgba
                ocrExecutor.submit {
                    try {
                        val ocrText = ocrEngine.recognizeTextInRegion(rgbaCopy, width, height, cx, cy, cw, ch)
                        if (ocrText.isNotBlank()) {
                            pendingOcrResult = mutableMapOf<String, Any>(
                                "format"         to "OCR",
                                "text"           to ocrText,
                                "confidence"     to detScore.toDouble(),
                                "isOcrFallback"  to true,
                                "source"         to "nanodet",
                                "boundingBox"    to mapOf(
                                    "x"      to bx1.toDouble(),
                                    "y"      to by1.toDouble(),
                                    "width"  to (bx2 - bx1).toDouble(),
                                    "height" to (by2 - by1).toDouble(),
                                ),
                                "cornerPoints" to emptyList<Any>(),
                            )
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Async OCR error: ${e.message}", e)
                    }
                }
            }
            results += buildUnknownResult(bx1, by1, bx2, by2, detScore, debug, debugCropBase64, allDebugLogs)
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
        val directLog = mutableListOf<String>()
        fun dlog(msg: String) { Log.d(TAG, "[DIRECT] $msg"); if (debug) directLog += "[DIRECT] $msg" }

        dlog("START fullFrame=${width}x${height}")

        val directBarcodes = ZXingNanoDetJNI.nativeDecodeBarcode(
            rgba, width, height, 0, 0, width, height, debug, false
        )

        // Extract JNI debug logs
        var dStartIdx = 0
        val jniLogLines = mutableListOf<String>()
        if (debug && directBarcodes.isNotEmpty() && directBarcodes[0].getOrNull(0) == "__log__") {
            dStartIdx = 1
            val logText = directBarcodes[0].getOrNull(1) ?: ""
            jniLogLines.addAll(logText.split("\n").filter { it.isNotBlank() })
            dlog("JNI produced ${jniLogLines.size} diagnostic lines")
        }
        val dBarcodes = if (dStartIdx > 0) directBarcodes.drop(dStartIdx) else directBarcodes.toList()
        dlog("rawBarcodes=${dBarcodes.size}")

        val filtered = if (enabledFormats != null) {
            dBarcodes.filter { bc -> enabledFormats.contains(bc.getOrNull(0) ?: "") }
        } else {
            dBarcodes
        }
        dlog("filteredBarcodes=${filtered.size} enabledFormats=${enabledFormats ?: "all"}")

        // Combine direct-level + JNI C++ diagnostic logs
        val allDebugLogs = if (debug) (directLog + jniLogLines) else null

        for (barcode in filtered) {
            val text = barcode.getOrNull(1) ?: ""
            if (text.isBlank()) continue
            dlog("result format=${barcode[0]} text=${text.take(40)}")
            val cornerPoints = (0..3).map { c ->
                mapOf(
                    "x" to barcode[2 + c * 2].toDouble(),
                    "y" to barcode[3 + c * 2].toDouble(),
                )
            }
            val resultMap = mutableMapOf<String, Any>(
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
            if (debug && allDebugLogs != null) resultMap["debugLogs"] = allDebugLogs
            results += resultMap
        }

        dlog("DONE results=${results.size}")
        return results
    }

    /**
     * Multi-scale center-crop fallback for distance scanning.
     * When full-frame ZXing fails, try progressively tighter center crops
     * where the upscale threshold will kick in to enlarge thin bars.
     */
    private fun decodeDirectZxingMultiScale(
        rgba: ByteArray, width: Int, height: Int,
        debug: Boolean, enabledFormats: Set<String>?,
    ): List<Map<String, Any>> {
        // First try full frame
        val fullResults = decodeDirectZxing(rgba, width, height, debug, enabledFormats)
        if (fullResults.isNotEmpty()) return fullResults

        // Try center crops at 50% then 33% — these are small enough for
        // ZXing's internal upscaler to activate (min dim <= upscaleThreshold).
        for (scale in doubleArrayOf(0.5, 0.33)) {
            val cropW = (width * scale).toInt()
            val cropH = (height * scale).toInt()
            val cropX = (width - cropW) / 2
            val cropY = (height - cropH) / 2
            Log.d(TAG, "[DIRECT_MULTISCALE] trying ${cropW}x${cropH} crop at ($cropX,$cropY)")

            val cropBarcodes = ZXingNanoDetJNI.nativeDecodeBarcode(
                rgba, width, height, cropX, cropY, cropW, cropH, debug, false
            )

            // Extract past log sentinel
            var startIdx = 0
            if (cropBarcodes.isNotEmpty() && cropBarcodes[0].getOrNull(0) == "__log__") startIdx = 1
            val barcodes = if (startIdx > 0) cropBarcodes.drop(startIdx) else cropBarcodes.toList()
            val filtered = if (enabledFormats != null) {
                barcodes.filter { bc -> enabledFormats.contains(bc.getOrNull(0) ?: "") }
            } else barcodes

            val results = mutableListOf<Map<String, Any>>()
            for (barcode in filtered) {
                val text = barcode.getOrNull(1) ?: ""
                if (text.isBlank()) continue
                // Map corner points from crop coords back to full frame coords
                val cornerPoints = (0..3).map { c ->
                    mapOf(
                        "x" to (barcode[2 + c * 2].toDouble() + cropX),
                        "y" to (barcode[3 + c * 2].toDouble() + cropY),
                    )
                }
                results += mutableMapOf<String, Any>(
                    "format"      to barcode[0],
                    "text"        to text,
                    "confidence"  to 0.5,
                    "source"      to "direct_multiscale",
                    "boundingBox" to mapOf(
                        "x"      to cropX.toDouble(),
                        "y"      to cropY.toDouble(),
                        "width"  to cropW.toDouble(),
                        "height" to cropH.toDouble(),
                    ),
                    "cornerPoints" to cornerPoints,
                )
            }
            if (results.isNotEmpty()) {
                Log.d(TAG, "[DIRECT_MULTISCALE] found ${results.size} at scale=$scale")
                return results
            }
        }

        return emptyList()
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