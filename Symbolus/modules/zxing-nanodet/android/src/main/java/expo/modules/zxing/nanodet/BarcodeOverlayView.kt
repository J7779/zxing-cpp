// SPDX-License-Identifier: Apache-2.0
// Native Android View that draws ZXing barcode bounding boxes and corner points.
// Rendered as an overlay on top of the camera preview.

package expo.modules.zxing.nanodet

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import android.view.View

class BarcodeOverlayView(context: Context) : View(context) {

    // Corner-point polygon
    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00FF00") // lime green
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    // Bounding-box rectangle
    private val bboxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF6600") // orange
        style = Paint.Style.STROKE
        strokeWidth = 2f
        pathEffect = DashPathEffect(floatArrayOf(12f, 6f), 0f)
    }

    // Corner vertices
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00FF00")
        style = Paint.Style.FILL
    }

    // Text label
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 28f
        typeface = Typeface.MONOSPACE
        setShadowLayer(4f, 1f, 1f, Color.BLACK)
    }

    // Semi-transparent background behind text
    private val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(160, 0, 0, 0)
        style = Paint.Style.FILL
    }

    // Coordinate label
    private val coordPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#AAFFAA")
        textSize = 20f
        typeface = Typeface.MONOSPACE
        setShadowLayer(3f, 1f, 1f, Color.BLACK)
    }

    // Show / hide the overlay
    var overlayVisible: Boolean = true
        set(value) { field = value; invalidate() }

    // Mirror X axis (front camera)
    var mirrorX: Boolean = false
        set(value) { field = value; invalidate() }

    private val listener: () -> Unit = { invalidate() }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        BarcodeOverlayManager.addListener(listener)
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        BarcodeOverlayManager.removeListener(listener)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!overlayVisible) return

        val barcodes = BarcodeOverlayManager.barcodes
        val fw = BarcodeOverlayManager.frameWidth.toFloat()
        val fh = BarcodeOverlayManager.frameHeight.toFloat()
        if (fw <= 0f || fh <= 0f) return

        val viewW = width.toFloat()
        val viewH = height.toFloat()
        if (viewW <= 0f || viewH <= 0f) return

        // Frame is landscape (e.g. 640×480), view is portrait (e.g. 400×700).
        // The camera preview center-crops, so we compute a scale-to-fill transform.
        // Frame coordinates: landscape (width>height typically).
        // View coordinates: portrait React Native view overlaid on camera.
        // VisionCamera on Android rotates the preview so the longest frame dim
        // maps to the longest view dim.  We need to swap frame w/h since the
        // sensor is landscape but the view may be portrait.

        val scaleX: Float
        val scaleY: Float
        val offsetX: Float
        val offsetY: Float

        // Determine if the frame needs rotation mapping (sensor landscape → view portrait)
        if ((fw > fh) != (viewW > viewH)) {
            // Frame is landscape, view is portrait → rotate 90° mapping
            // Frame x → view y, frame y → view x (mirrored)
            scaleX = viewW / fh
            scaleY = viewH / fw
            offsetX = 0f
            offsetY = 0f
        } else {
            scaleX = viewW / fw
            scaleY = viewH / fh
            offsetX = 0f
            offsetY = 0f
        }

        val rotateMapping = (fw > fh) != (viewW > viewH)

        fun mapX(fx: Float, fy: Float): Float {
            val x = if (rotateMapping) (fh - fy) * scaleX else fx * scaleX
            return if (mirrorX) viewW - x + offsetX else x + offsetX
        }
        fun mapY(fx: Float, fy: Float): Float {
            return if (rotateMapping) fx * scaleY + offsetY else fy * scaleY + offsetY
        }

        for (bc in barcodes) {
            // Draw corner-point polygon
            if (bc.corners.size == 4) {
                val path = Path()
                val (fx0, fy0) = bc.corners[0]
                path.moveTo(mapX(fx0, fy0), mapY(fx0, fy0))
                for (i in 1..3) {
                    val (fx, fy) = bc.corners[i]
                    path.lineTo(mapX(fx, fy), mapY(fx, fy))
                }
                path.close()
                canvas.drawPath(path, cornerPaint)

                // Draw corner dots with coordinate labels
                for (i in bc.corners.indices) {
                    val (fx, fy) = bc.corners[i]
                    val vx = mapX(fx, fy)
                    val vy = mapY(fx, fy)
                    canvas.drawCircle(vx, vy, 6f, dotPaint)
                    // Show raw frame coordinates at each corner
                    canvas.drawText("(${fx.toInt()},${fy.toInt()})", vx + 8f, vy - 4f, coordPaint)
                }
            }

            // Draw bounding box rectangle (dashed orange)
            val bx = bc.bboxX
            val by = bc.bboxY
            val bw = bc.bboxW
            val bh = bc.bboxH
            if (bw > 0 && bh > 0) {
                val left   = mapX(bx, by)
                val top    = mapY(bx, by)
                val right  = mapX(bx + bw, by + bh)
                val bottom = mapY(bx + bw, by + bh)
                canvas.drawRect(
                    minOf(left, right), minOf(top, bottom),
                    maxOf(left, right), maxOf(top, bottom),
                    bboxPaint,
                )
            }

            // Draw text label with format and decoded text
            if (bc.corners.isNotEmpty()) {
                val (lx, ly) = bc.corners[0]
                val labelX = mapX(lx, ly)
                val labelY = mapY(lx, ly) - 12f

                val label = "${bc.format}: ${bc.text.take(30)}"
                val textWidth = textPaint.measureText(label)
                canvas.drawRoundRect(
                    RectF(labelX - 4f, labelY - 26f, labelX + textWidth + 4f, labelY + 4f),
                    4f, 4f, textBgPaint,
                )
                canvas.drawText(label, labelX, labelY, textPaint)
            }
        }
    }
}
