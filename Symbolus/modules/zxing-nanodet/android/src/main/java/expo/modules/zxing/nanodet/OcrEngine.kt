// SPDX-License-Identifier: Apache-2.0
// OcrEngine.kt
//
// PP-OCRv5 mobile OCR fallback for barcodes that ZXing cannot decode.
//
// Pipeline:
//   1. (Optional) Text detection  — ppocr_v5_mobile_det.onnx (DBNet)
//   2. Text recognition           — ppocr_v5_mobile_rec.onnx (CRNN + CTC)
//
// Models are pre-converted ONNX from bukuroo/PPOCRv5-ONNX on HuggingFace and
// placed in src/main/assets/:
//   ppocr_v5_mobile_det.onnx  — PP-OCRv5 mobile text detector  (4.6 MB)
//   ppocr_v5_mobile_rec.onnx  — PP-OCRv5 mobile recogniser     (16 MB)
//   ppocr_v5_dict.txt         — 18383-entry UTF-8 character dictionary
//
// Model I/O (verified with onnxruntime):
//   det  input  x              [1, 3, H, W]  (dynamic H/W, multiples of 32)
//   det  output fetch_name_0   [1, 1, H, W]  probability map
//   rec  input  x              [1, 3, 48, W] (H fixed = 48, W dynamic)
//   rec  output fetch_name_0   [1, T, 18385] logits — CTC decode:
//                                 0         = CTC blank
//                                 1..18383  = dict[index-1] (ppocr_v5_dict.txt)
//                                 18384     = space " "
//
// If only the rec model is present, detection is skipped and the whole expanded
// region below the NanoDet bounding box is treated as one text line.

package expo.modules.zxing.nanodet

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer

private const val OCR_TAG = "PP-OCRv5"
private const val DET_MODEL_ASSET  = "ppocr_v5_mobile_det.onnx"
private const val REC_MODEL_ASSET  = "ppocr_v5_mobile_rec.onnx"
private const val DICT_ASSET       = "ppocr_v5_dict.txt"
// CTC class layout: 0=blank, 1..N=dict chars, N+1=space
private const val CTC_BLANK        = 0
private const val CTC_SPACE_OFFSET = 1  // spaceclassIndex = dictSize + CTC_SPACE_OFFSET

/**
 * PP-OCRv5 inference engine.
 *
 * Constructed once inside [ZXingNanoDetPlugin] and reused for every frame
 * that ZXing fails to decode. Thread-safe (ORT sessions are thread-safe).
 */
class OcrEngine(
    context: Context,
    private val ortEnv: OrtEnvironment,
) {
    private val detSession: OrtSession? = loadSession(context, DET_MODEL_ASSET)
    private val recSession: OrtSession? = loadSession(context, REC_MODEL_ASSET)

    // Dictionary loaded from ppocr_v5_dict.txt.
    // dict[i] is the character string for class index (i+1).
    // Class 0 = CTC blank; class dictSize+1 = space.
    private val dict: Array<String> = loadDict(context)

    /** True when the recognition model is available; detection is optional. */
    val isAvailable: Boolean get() = recSession != null

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Attempt to read text from the region surrounding [cx],[cy],[cw],[ch].
     *
     * The crop is expanded in the direction that corresponds to "below the barcode"
     * in the display frame (where HRI digits live in a standard 1D barcode).
     *
     * Coordinate geometry (landscape sensor, phone portrait):
     *   90° CW rotation: sensor_col → display_row  (sensor x → display y top→bottom)
     * Therefore "below barcode in display" = HIGHER sensor.x = expand cx+cw rightward.
     *
     * For portrait sensor frames (phone landscape) HRI is simply below (higher sensor.y).
     *
     * @return Recognized text string, or empty string on failure / model absent.
     */
    fun recognizeTextInRegion(
        rgba: ByteArray,
        frameW: Int,
        frameH: Int,
        cx: Int, cy: Int, cw: Int, ch: Int,
    ): String {
        val rec = recSession ?: return ""
        return try {
            val needsRotation = frameW > frameH
            // The cx/cy/cw/ch passed here already include the 30% NanoDet padding from the
            // call-site, which is more than enough margin to enclose the HRI strip that lives
            // at the bottom ~20% of the barcode symbol.  Do NOT expand further: extra columns
            // (landscape sensor) or extra rows (portrait sensor) would push the 28%-bottom
            // HRI slice entirely into empty camera space past the barcode.
            Log.d(OCR_TAG, "[OCR_REGION] frame=${frameW}x${frameH} needsRotation=$needsRotation" +
                " barcode=($cx,$cy ${cw}x$ch)")
            recognizeRegion(rgba, frameW, frameH, cx, cy, cw, ch, needsRotation, rec)
        } catch (e: Exception) {
            Log.e(OCR_TAG, "OCR error: ${e.message}", e)
            ""
        }
    }

    // ── Full pipeline: detection + recognition ────────────────────────────

    private fun recognizeWithDetection(
        rgba: ByteArray,
        frameW: Int, frameH: Int,
        cx: Int, cy: Int, cw: Int, ch: Int,
        needsRotation: Boolean,
        det: OrtSession,
        rec: OrtSession,
    ): String {
        val (pixels, cropW, cropH) = cropAndRotateRgb(rgba, frameW, frameH, cx, cy, cw, ch, needsRotation)
        if (cropW == 0 || cropH == 0) return ""

        // Resize so that the longer edge ≤ 960, aligned to 32-pixel boundary
        val maxDim = 960
        val rawScale = minOf(maxDim.toFloat() / maxOf(cropW, cropH), 1f)
        val detW = (((cropW * rawScale) / 32).toInt().coerceAtLeast(1) * 32).coerceAtMost(maxDim)
        val detH = (((cropH * rawScale) / 32).toInt().coerceAtLeast(1) * 32).coerceAtMost(maxDim)
        val resized = bilinearResize(pixels, cropW, cropH, detW, detH)

        // Normalize with ImageNet mean / std: (v/255 - mean) / std, RGB channel order
        val tensor = FloatArray(3 * detH * detW)
        val means = floatArrayOf(0.485f, 0.456f, 0.406f)
        val stds  = floatArrayOf(0.229f, 0.224f, 0.225f)
        for (row in 0 until detH) {
            for (col in 0 until detW) {
                val src = (row * detW + col) * 3
                for (c in 0 until 3) {
                    val v = (resized[src + c].toInt() and 0xFF) / 255f
                    tensor[c * detH * detW + row * detW + col] = (v - means[c]) / stds[c]
                }
            }
        }

        // Run detection model
        val detInput = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(tensor),
            longArrayOf(1, 3, detH.toLong(), detW.toLong()),
        )
        val detOuts = det.run(mapOf(det.inputNames.first() to detInput))
        detInput.close()

        val probTensor = detOuts.get(det.outputNames.first()).get() as OnnxTensor
        val probBuf = probTensor.floatBuffer
        val probMap = FloatArray(probBuf.remaining()).also { probBuf.get(it) }
        probTensor.close()
        detOuts.close()

        // DB post-process: find text region bounding boxes
        val textBoxes = extractTextBoxes(probMap, detH, detW, threshold = 0.3f, rawScale, cropW, cropH)

        if (textBoxes.isEmpty()) {
            // No text detected — fall back to recognizing the whole crop
            return recognizeRegion(rgba, frameW, frameH, cx, cy, cw, ch, needsRotation, rec)
        }

        // Recognize text in each detected box and join
        return textBoxes.mapNotNull { (bx, by, bw, bh) ->
            val absCx = cx + bx; val absCy = cy + by
            recognizeRegion(rgba, frameW, frameH, absCx, absCy, bw, bh, needsRotation, rec)
                .takeIf { it.isNotBlank() }
        }.joinToString(" ")
    }

    // ── Recognition-only path ─────────────────────────────────────────────

    private fun recognizeRegion(
        rgba: ByteArray,
        frameW: Int, frameH: Int,
        cx: Int, cy: Int, cw: Int, ch: Int,
        needsRotation: Boolean,
        rec: OrtSession,
    ): String {
        val (pixels, cropW, cropH) = cropAndRotateRgb(rgba, frameW, frameH, cx, cy, cw, ch, needsRotation)
        if (cropW == 0 || cropH == 0) return ""

        // After 90° CW rotation the barcode stands upright (portrait orientation):
        // the bars fill most of the height and the narrow HRI digit strip sits at
        // the very bottom (~20-25 %). Feeding the entire image to the CRNN gives
        // almost no timesteps (seqLen≈6 for 49px wide) and all-blank CTC output.
        //
        // Strategy:
        //  • If the crop is "tall" relative to width (aspect < 2.0 wide) it is the
        //    full barcode symbol; slice off the bottom 28 % as the HRI strip.
        //  • Otherwise the caller already provided a text-line-shaped crop; use it as-is.
        val (stripPixels, stripW, stripH) = if (cropW < cropH * 2) {
            // Looks like a full barcode column. Take bottom 28 % = HRI strip.
            val stripStart = (cropH * 0.72f).toInt().coerceAtLeast(1)
            val sh = (cropH - stripStart).coerceAtLeast(1)
            val strip = ByteArray(cropW * sh * 3)
            System.arraycopy(pixels, stripStart * cropW * 3, strip, 0, cropW * sh * 3)
            Log.d(OCR_TAG, "[HRI_STRIP] full barcode ${cropW}x${cropH} → bottom strip ${cropW}x${sh} (rows ${stripStart}..${cropH})")
            Triple(strip, cropW, sh)
        } else {
            // Already a reasonable text-line aspect ratio; use whole crop.
            Triple(pixels, cropW, cropH)
        }

        // Resize to 48 px tall, proportional width.
        // Minimum 8: the rec model's internal conv layers have a 4× horizontal stride;
        // widths below 8 collapse to 0 internally → OrtException: Invalid input shape {1,0}.
        val recH = 48
        val recW = ((stripW.toFloat() / stripH) * recH).toInt().coerceIn(8, 1200)
        Log.d(OCR_TAG, "[REC_INPUT] sensorCrop=($cx,$cy ${cw}x$ch) rotated=${cropW}x${cropH} strip=${stripW}x${stripH} recInput=${recW}x${recH}")
        val resized = bilinearResize(stripPixels, stripW, stripH, recW, recH)

        // Normalize: (v/255 - 0.5) / 0.5
        val tensor = FloatArray(3 * recH * recW)
        for (row in 0 until recH) {
            for (col in 0 until recW) {
                val src = (row * recW + col) * 3
                for (c in 0 until 3) {
                    val v = (resized[src + c].toInt() and 0xFF) / 255f
                    tensor[c * recH * recW + row * recW + col] = (v - 0.5f) / 0.5f
                }
            }
        }

        val recInput = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(tensor),
            longArrayOf(1, 3, recH.toLong(), recW.toLong()),
        )
        val recOuts = rec.run(mapOf(rec.inputNames.first() to recInput))
        recInput.close()

        val outTensor = recOuts.get(rec.outputNames.first()).get() as OnnxTensor
        val shape = outTensor.info.shape
        val logitsBuf = outTensor.floatBuffer
        val logits = FloatArray(logitsBuf.remaining()).also { logitsBuf.get(it) }
        outTensor.close()
        recOuts.close()

        // shape: [1, seqLen, numClasses]
        val seqLen     = shape[1].toInt()
        val numClasses = shape[2].toInt()
        val result = ctcDecode(logits, seqLen, numClasses)

        // Debug: if empty, log the first 20 argmax tokens so we can see what the model sees
        if (result.isEmpty() && seqLen > 0) {
            val spaceIdx = dict.size + 1
            val sample = (0 until minOf(seqLen, 30)).joinToString("") { t ->
                val base = t * numClasses
                var mi = 0; var mv = logits[base]
                for (c in 1 until numClasses) { val v = logits[base + c]; if (v > mv) { mv = v; mi = c } }
                when {
                    mi == CTC_BLANK       -> "_"
                    mi == spaceIdx        -> " "
                    mi in 1..dict.size    -> dict[mi - 1]
                    else                  -> "?"
                }
            }
            Log.d(OCR_TAG, "[CTC_DEBUG] seqLen=$seqLen numClasses=$numClasses empty result, token sample: '$sample'")
        } else {
            Log.d(OCR_TAG, "[REC_RESULT] seqLen=$seqLen result='$result'")
        }
        return result
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /**
     * Crop [cx,cy,cw,ch] from an RGBA frame and optionally rotate 90° CW
     * (matching the ZXing preprocessing path for landscape sensor frames).
     *
     * Returns (RGB byte array, outputWidth, outputHeight).
     */
    private fun cropAndRotateRgb(
        rgba: ByteArray,
        frameW: Int, frameH: Int,
        cx: Int, cy: Int, cw: Int, ch: Int,
        needsRotation: Boolean,
    ): Triple<ByteArray, Int, Int> {
        val safeCx = cx.coerceIn(0, (frameW - 1).coerceAtLeast(0))
        val safeCy = cy.coerceIn(0, (frameH - 1).coerceAtLeast(0))
        val safeCw = cw.coerceIn(0, frameW - safeCx)
        val safeCh = ch.coerceIn(0, frameH - safeCy)
        if (safeCw == 0 || safeCh == 0) return Triple(ByteArray(0), 0, 0)

        return if (!needsRotation) {
            val rgb = ByteArray(safeCw * safeCh * 3)
            for (row in 0 until safeCh) {
                for (col in 0 until safeCw) {
                    val srcIdx = ((safeCy + row) * frameW + (safeCx + col)) * 4
                    val dstIdx = (row * safeCw + col) * 3
                    rgb[dstIdx]     = rgba[srcIdx]
                    rgb[dstIdx + 1] = rgba[srcIdx + 1]
                    rgb[dstIdx + 2] = rgba[srcIdx + 2]
                }
            }
            Triple(rgb, safeCw, safeCh)
        } else {
            // 90° CW: source (row, col) → dest (col, safeCh-1-row)
            val rotW = safeCh
            val rotH = safeCw
            val rgb = ByteArray(rotW * rotH * 3)
            for (row in 0 until safeCh) {
                for (col in 0 until safeCw) {
                    val srcIdx = ((safeCy + row) * frameW + (safeCx + col)) * 4
                    val newRow = col
                    val newCol = safeCh - 1 - row
                    val dstIdx = (newRow * rotW + newCol) * 3
                    rgb[dstIdx]     = rgba[srcIdx]
                    rgb[dstIdx + 1] = rgba[srcIdx + 1]
                    rgb[dstIdx + 2] = rgba[srcIdx + 2]
                }
            }
            Triple(rgb, rotW, rotH)
        }
    }

    /** Bilinear resize of an RGB (3ch interleaved) byte array. */
    private fun bilinearResize(src: ByteArray, srcW: Int, srcH: Int, dstW: Int, dstH: Int): ByteArray {
        val dst = ByteArray(dstW * dstH * 3)
        val xRatio = srcW.toFloat() / dstW
        val yRatio = srcH.toFloat() / dstH
        for (row in 0 until dstH) {
            val srcY = row * yRatio
            val y0 = srcY.toInt().coerceIn(0, srcH - 1)
            val y1 = (y0 + 1).coerceIn(0, srcH - 1)
            val yFrac = srcY - y0
            for (col in 0 until dstW) {
                val srcX = col * xRatio
                val x0 = srcX.toInt().coerceIn(0, srcW - 1)
                val x1 = (x0 + 1).coerceIn(0, srcW - 1)
                val xFrac = srcX - x0
                val dstIdx = (row * dstW + col) * 3
                for (c in 0 until 3) {
                    val p00 = (src[(y0 * srcW + x0) * 3 + c].toInt() and 0xFF).toFloat()
                    val p10 = (src[(y0 * srcW + x1) * 3 + c].toInt() and 0xFF).toFloat()
                    val p01 = (src[(y1 * srcW + x0) * 3 + c].toInt() and 0xFF).toFloat()
                    val p11 = (src[(y1 * srcW + x1) * 3 + c].toInt() and 0xFF).toFloat()
                    val v = p00 * (1 - xFrac) * (1 - yFrac) +
                            p10 * xFrac       * (1 - yFrac) +
                            p01 * (1 - xFrac) * yFrac +
                            p11 * xFrac       * yFrac
                    dst[dstIdx + c] = v.toInt().coerceIn(0, 255).toByte()
                }
            }
        }
        return dst
    }

    /**
     * DB text detection post-processing.
     *
     * Thresholds the probability map, finds connected components (BFS),
     * and returns bounding boxes in original (pre-scale) crop coordinates.
     *
     * @return List of [x, y, w, h] bounding boxes.
     */
    private fun extractTextBoxes(
        probMap: FloatArray,
        mapH: Int, mapW: Int,
        threshold: Float,
        scale: Float,
        origW: Int, origH: Int,
    ): List<IntArray> {
        val binary   = Array(mapH) { row -> BooleanArray(mapW) { col -> probMap[row * mapW + col] > threshold } }
        val visited  = Array(mapH) { BooleanArray(mapW) }
        val boxes    = mutableListOf<IntArray>()

        for (startRow in 0 until mapH) {
            for (startCol in 0 until mapW) {
                if (!binary[startRow][startCol] || visited[startRow][startCol]) continue

                // BFS connected component
                var minX = startCol; var maxX = startCol
                var minY = startRow; var maxY = startRow
                val queue = ArrayDeque<Long>()  // encode (row, col) as single Long for efficiency
                queue.add(startRow.toLong() shl 32 or startCol.toLong())
                visited[startRow][startCol] = true

                while (queue.isNotEmpty()) {
                    val encoded = queue.removeFirst()
                    val r = (encoded ushr 32).toInt()
                    val c = (encoded and 0xFFFFFFFFL).toInt()
                    if (c < minX) minX = c; if (c > maxX) maxX = c
                    if (r < minY) minY = r; if (r > maxY) maxY = r

                    for ((dr, dc) in DIRS) {
                        val nr = r + dr; val nc = c + dc
                        if (nr in 0 until mapH && nc in 0 until mapW &&
                            binary[nr][nc] && !visited[nr][nc]
                        ) {
                            visited[nr][nc] = true
                            queue.add(nr.toLong() shl 32 or nc.toLong())
                        }
                    }
                }

                val bw = maxX - minX + 1
                val bh = maxY - minY + 1
                if (bw < 3 || bh < 3) continue   // ignore noise

                // Map back to original crop coordinates
                val ox = (minX / scale).toInt().coerceIn(0, origW - 1)
                val oy = (minY / scale).toInt().coerceIn(0, origH - 1)
                val ow = ((bw  / scale).toInt()).coerceIn(1, origW - ox)
                val oh = ((bh  / scale).toInt()).coerceIn(1, origH - oy)
                boxes.add(intArrayOf(ox, oy, ow, oh))
            }
        }
        return boxes
    }

    /**
     * Greedy CTC decoder: argmax per timestep, collapse repeated tokens,
     * remove blank (class 0).
     *
     * Class mapping (matches ppocr_v5_dict.txt convention):
     *   0            → CTC blank (skip)
     *   1..dictSize  → dict[classIdx - 1]
     *   dictSize+1   → " " (space)
     */
    private fun ctcDecode(logits: FloatArray, seqLen: Int, numClasses: Int): String {
        val sb = StringBuilder()
        var prevIdx = -1
        val spaceIdx = dict.size + 1  // dictSize + 1
        for (t in 0 until seqLen) {
            val base = t * numClasses
            var maxIdx = 0
            var maxVal = logits[base]
            for (c in 1 until numClasses) {
                val v = logits[base + c]
                if (v > maxVal) { maxVal = v; maxIdx = c }
            }
            if (maxIdx != CTC_BLANK && maxIdx != prevIdx) {
                when {
                    maxIdx == spaceIdx                        -> sb.append(' ')
                    maxIdx in 1..dict.size                   -> sb.append(dict[maxIdx - 1])
                    // ignore any out-of-range class
                }
            }
            prevIdx = maxIdx
        }
        return sb.toString().trim()
    }

    private fun loadSession(context: Context, assetName: String): OrtSession? = try {
        val bytes = context.assets.open(assetName).readBytes()
        val opts  = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        ortEnv.createSession(bytes, opts).also {
            Log.i(OCR_TAG, "Loaded $assetName  inputs=${it.inputNames}  outputs=${it.outputNames}")
        }
    } catch (e: Exception) {
        Log.w(OCR_TAG, "OCR model not found: $assetName — ${e.message}. " +
            "Place ONNX models in src/main/assets/ to enable PaddleOCR fallback.")
        null
    }

    /**
     * Loads ppocr_v5_dict.txt from assets.
     * Each line = one character/string entry.
     * Returns an array where dict[i] maps to class index i+1.
     */
    private fun loadDict(context: Context): Array<String> = try {
        context.assets.open(DICT_ASSET)
            .bufferedReader(Charsets.UTF_8)
            .readLines()
            .toTypedArray()
            .also { Log.i(OCR_TAG, "Loaded dict: ${it.size} entries") }
    } catch (e: Exception) {
        Log.w(OCR_TAG, "Dict not found ($DICT_ASSET): ${e.message}")
        emptyArray()
    }

    companion object {
        private val DIRS = arrayOf(intArrayOf(-1, 0), intArrayOf(1, 0), intArrayOf(0, -1), intArrayOf(0, 1))
    }
}
