// SPDX-License-Identifier: Apache-2.0
// Camera2ISPModule.kt
//
// React Native native module that applies Camera2 ISP (Image Signal Processor)
// settings to the active CameraX camera session used by VisionCamera.
//
// Exposed settings match what Scandit configures internally:
//   • Noise Reduction Mode  (OFF / FAST / HIGH_QUALITY)
//   • Edge Enhancement Mode (OFF / FAST / HIGH_QUALITY)
//   • Tonemap Mode          (FAST / HIGH_QUALITY + optional GAMMA_2_2 curve)
//   • AE Exposure Compensation (bias for barcode-friendly brightness)
//   • AE/AF Region of Interest (metering region for scan area)

package expo.modules.zxing.nanodet

import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.TonemapCurve
import android.os.Build
import android.util.Log
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.CaptureRequestOptions
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableMap
import java.util.concurrent.Executors

private const val TAG = "Camera2ISPModule"

class Camera2ISPModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String = "Camera2ISP"

    // Cached reference to the CameraX camera — resolved on first applyISPSettings call
    private var lastCamera: androidx.camera.core.Camera? = null

    /**
     * Apply Camera2 ISP settings to the active CameraX camera.
     *
     * @param config ReadableMap with optional keys:
     *   - noiseReduction: "off" | "fast" | "high_quality"  (default: "fast")
     *   - edgeEnhancement: "off" | "fast" | "high_quality" (default: "fast")
     *   - tonemap: "none" | "fast" | "high_quality" | "gamma22" (default: "fast")
     *   - exposureCompensation: int (EV steps, typically -6..+6, default: 0)
     *   - colorCorrection: "transform_matrix" | "fast" | "high_quality" (default: "fast")
     */
    @ReactMethod
    fun applyISPSettings(config: ReadableMap, promise: Promise) {
        try {
            val camera = findActiveCamera()
            if (camera == null) {
                promise.reject("NO_CAMERA", "No active CameraX camera found")
                return
            }

            val camera2Control = Camera2CameraControl.from(camera.cameraControl)

            val builder = CaptureRequestOptions.Builder()

            // --- Noise Reduction ---
            if (config.hasKey("noiseReduction")) {
                val mode = when (config.getString("noiseReduction")) {
                    "off" -> CaptureRequest.NOISE_REDUCTION_MODE_OFF
                    "fast" -> CaptureRequest.NOISE_REDUCTION_MODE_FAST
                    "high_quality" -> CaptureRequest.NOISE_REDUCTION_MODE_HIGH_QUALITY
                    "minimal" -> CaptureRequest.NOISE_REDUCTION_MODE_MINIMAL
                    "zero_shutter_lag" -> if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M)
                        CaptureRequest.NOISE_REDUCTION_MODE_ZERO_SHUTTER_LAG
                    else CaptureRequest.NOISE_REDUCTION_MODE_FAST
                    else -> CaptureRequest.NOISE_REDUCTION_MODE_FAST
                }
                builder.setCaptureRequestOption(CaptureRequest.NOISE_REDUCTION_MODE, mode)
                Log.d(TAG, "Noise Reduction: $mode")
            }

            // --- Edge Enhancement ---
            if (config.hasKey("edgeEnhancement")) {
                val mode = when (config.getString("edgeEnhancement")) {
                    "off" -> CaptureRequest.EDGE_MODE_OFF
                    "fast" -> CaptureRequest.EDGE_MODE_FAST
                    "high_quality" -> CaptureRequest.EDGE_MODE_HIGH_QUALITY
                    "zero_shutter_lag" -> if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M)
                        CaptureRequest.EDGE_MODE_ZERO_SHUTTER_LAG
                    else CaptureRequest.EDGE_MODE_FAST
                    else -> CaptureRequest.EDGE_MODE_FAST
                }
                builder.setCaptureRequestOption(CaptureRequest.EDGE_MODE, mode)
                Log.d(TAG, "Edge Enhancement: $mode")
            }

            // --- Tonemap ---
            if (config.hasKey("tonemap")) {
                val tonemapStr = config.getString("tonemap")
                when (tonemapStr) {
                    "fast" -> {
                        builder.setCaptureRequestOption(
                            CaptureRequest.TONEMAP_MODE,
                            CaptureRequest.TONEMAP_MODE_FAST
                        )
                    }
                    "high_quality" -> {
                        builder.setCaptureRequestOption(
                            CaptureRequest.TONEMAP_MODE,
                            CaptureRequest.TONEMAP_MODE_HIGH_QUALITY
                        )
                    }
                    "gamma22" -> {
                        // Apply gamma 2.2 curve for consistent contrast
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                            builder.setCaptureRequestOption(
                                CaptureRequest.TONEMAP_MODE,
                                CaptureRequest.TONEMAP_MODE_GAMMA_VALUE
                            )
                            builder.setCaptureRequestOption(
                                CaptureRequest.TONEMAP_GAMMA, 2.2f
                            )
                        } else {
                            // Fallback: use contrast curve for pre-Marshmallow
                            builder.setCaptureRequestOption(
                                CaptureRequest.TONEMAP_MODE,
                                CaptureRequest.TONEMAP_MODE_CONTRAST_CURVE
                            )
                            val gamma22Curve = buildGamma22Curve()
                            builder.setCaptureRequestOption(
                                CaptureRequest.TONEMAP_CURVE, gamma22Curve
                            )
                        }
                    }
                    "srgb" -> {
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                            builder.setCaptureRequestOption(
                                CaptureRequest.TONEMAP_MODE,
                                CaptureRequest.TONEMAP_MODE_PRESET_CURVE
                            )
                            builder.setCaptureRequestOption(
                                CaptureRequest.TONEMAP_PRESET_CURVE,
                                CaptureRequest.TONEMAP_PRESET_CURVE_SRGB
                            )
                        } else {
                            builder.setCaptureRequestOption(
                                CaptureRequest.TONEMAP_MODE,
                                CaptureRequest.TONEMAP_MODE_FAST
                            )
                        }
                    }
                    "none", "off" -> {
                        // Linear / no tonemap
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                            builder.setCaptureRequestOption(
                                CaptureRequest.TONEMAP_MODE,
                                CaptureRequest.TONEMAP_MODE_GAMMA_VALUE
                            )
                            builder.setCaptureRequestOption(
                                CaptureRequest.TONEMAP_GAMMA, 1.0f
                            )
                        }
                    }
                }
                Log.d(TAG, "Tonemap: $tonemapStr")
            }

            // --- Exposure Compensation ---
            if (config.hasKey("exposureCompensation")) {
                val ev = config.getInt("exposureCompensation")
                builder.setCaptureRequestOption(
                    CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, ev
                )
                Log.d(TAG, "Exposure Compensation: $ev EV steps")
            }

            // --- Color Correction ---
            if (config.hasKey("colorCorrection")) {
                val mode = when (config.getString("colorCorrection")) {
                    "transform_matrix" -> CaptureRequest.COLOR_CORRECTION_MODE_TRANSFORM_MATRIX
                    "fast" -> CaptureRequest.COLOR_CORRECTION_MODE_FAST
                    "high_quality" -> CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY
                    else -> CaptureRequest.COLOR_CORRECTION_MODE_FAST
                }
                builder.setCaptureRequestOption(CaptureRequest.COLOR_CORRECTION_MODE, mode)
                Log.d(TAG, "Color Correction: $mode")
            }

            // --- Shading Mode (lens shading correction) ---
            if (config.hasKey("shadingMode")) {
                val mode = when (config.getString("shadingMode")) {
                    "off" -> CaptureRequest.SHADING_MODE_OFF
                    "fast" -> CaptureRequest.SHADING_MODE_FAST
                    "high_quality" -> CaptureRequest.SHADING_MODE_HIGH_QUALITY
                    else -> CaptureRequest.SHADING_MODE_FAST
                }
                builder.setCaptureRequestOption(CaptureRequest.SHADING_MODE, mode)
                Log.d(TAG, "Shading Mode: $mode")
            }

            // --- Hot Pixel Mode ---
            if (config.hasKey("hotPixelMode")) {
                val mode = when (config.getString("hotPixelMode")) {
                    "off" -> CaptureRequest.HOT_PIXEL_MODE_OFF
                    "fast" -> CaptureRequest.HOT_PIXEL_MODE_FAST
                    "high_quality" -> CaptureRequest.HOT_PIXEL_MODE_HIGH_QUALITY
                    else -> CaptureRequest.HOT_PIXEL_MODE_FAST
                }
                builder.setCaptureRequestOption(CaptureRequest.HOT_PIXEL_MODE, mode)
                Log.d(TAG, "Hot Pixel Mode: $mode")
            }

            // Apply all options at once
            val future = camera2Control.addCaptureRequestOptions(builder.build())
            future.addListener({
                try {
                    future.get()
                    Log.i(TAG, "ISP settings applied successfully")
                    promise.resolve(true)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to apply ISP settings: ${e.message}", e)
                    promise.reject("ISP_ERROR", "Failed to apply: ${e.message}", e)
                }
            }, Executors.newSingleThreadExecutor())

        } catch (e: Exception) {
            Log.e(TAG, "applyISPSettings error: ${e.message}", e)
            promise.reject("ISP_ERROR", e.message, e)
        }
    }

    /**
     * Query what Camera2 capabilities are available on this device.
     * Returns a map of supported ISP features.
     */
    @ReactMethod
    fun getISPCapabilities(promise: Promise) {
        try {
            val cameraManager = reactApplicationContext
                .getSystemService(android.content.Context.CAMERA_SERVICE) as CameraManager

            val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
                val chars = cameraManager.getCameraCharacteristics(id)
                chars.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
            }

            if (cameraId == null) {
                promise.reject("NO_CAMERA", "No back-facing camera found")
                return
            }

            val chars = cameraManager.getCameraCharacteristics(cameraId)
            val result = com.facebook.react.bridge.Arguments.createMap()

            // Noise Reduction modes
            val nrModes = chars.get(CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES)
            val nrArray = com.facebook.react.bridge.Arguments.createArray()
            nrModes?.forEach { mode ->
                nrArray.pushString(when (mode) {
                    CaptureRequest.NOISE_REDUCTION_MODE_OFF -> "off"
                    CaptureRequest.NOISE_REDUCTION_MODE_FAST -> "fast"
                    CaptureRequest.NOISE_REDUCTION_MODE_HIGH_QUALITY -> "high_quality"
                    CaptureRequest.NOISE_REDUCTION_MODE_MINIMAL -> "minimal"
                    5 -> "zero_shutter_lag"
                    else -> "unknown_$mode"
                })
            }
            result.putArray("noiseReduction", nrArray)

            // Edge Enhancement modes
            val edgeModes = chars.get(CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES)
            val edgeArray = com.facebook.react.bridge.Arguments.createArray()
            edgeModes?.forEach { mode ->
                edgeArray.pushString(when (mode) {
                    CaptureRequest.EDGE_MODE_OFF -> "off"
                    CaptureRequest.EDGE_MODE_FAST -> "fast"
                    CaptureRequest.EDGE_MODE_HIGH_QUALITY -> "high_quality"
                    3 -> "zero_shutter_lag"
                    else -> "unknown_$mode"
                })
            }
            result.putArray("edgeEnhancement", edgeArray)

            // Tonemap modes
            val tonemapModes = chars.get(CameraCharacteristics.TONEMAP_AVAILABLE_TONE_MAP_MODES)
            val tonemapArray = com.facebook.react.bridge.Arguments.createArray()
            tonemapModes?.forEach { mode ->
                tonemapArray.pushString(when (mode) {
                    CaptureRequest.TONEMAP_MODE_CONTRAST_CURVE -> "contrast_curve"
                    CaptureRequest.TONEMAP_MODE_FAST -> "fast"
                    CaptureRequest.TONEMAP_MODE_HIGH_QUALITY -> "high_quality"
                    3 -> "gamma_value"
                    4 -> "preset_curve"
                    else -> "unknown_$mode"
                })
            }
            result.putArray("tonemap", tonemapArray)

            // Exposure compensation range
            val aeRange = chars.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)
            if (aeRange != null) {
                val aeMap = com.facebook.react.bridge.Arguments.createMap()
                aeMap.putInt("min", aeRange.lower)
                aeMap.putInt("max", aeRange.upper)
                val step = chars.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_STEP)
                if (step != null) {
                    aeMap.putDouble("step", step.toDouble())
                }
                result.putMap("exposureCompensation", aeMap)
            }

            // Hardware level
            val hwLevel = chars.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
            result.putString("hardwareLevel", when (hwLevel) {
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY"
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "EXTERNAL"
                else -> "UNKNOWN"
            })

            promise.resolve(result)
        } catch (e: Exception) {
            Log.e(TAG, "getISPCapabilities error: ${e.message}", e)
            promise.reject("ISP_ERROR", e.message, e)
        }
    }

    /**
     * Reset all ISP settings to Camera2 defaults (let CameraX / auto choose).
     */
    @ReactMethod
    fun resetISPSettings(promise: Promise) {
        try {
            val camera = findActiveCamera()
            if (camera == null) {
                promise.reject("NO_CAMERA", "No active CameraX camera found")
                return
            }
            val camera2Control = Camera2CameraControl.from(camera.cameraControl)

            // Clear all custom capture request options
            val emptyOptions = CaptureRequestOptions.Builder().build()
            val future = camera2Control.clearCaptureRequestOptions()
            future.addListener({
                try {
                    future.get()
                    Log.i(TAG, "ISP settings reset to defaults")
                    promise.resolve(true)
                } catch (e: Exception) {
                    promise.reject("ISP_ERROR", "Failed to reset: ${e.message}", e)
                }
            }, Executors.newSingleThreadExecutor())
        } catch (e: Exception) {
            promise.reject("ISP_ERROR", e.message, e)
        }
    }

    // -- Internal: find the active CameraX Camera instance ----------------------

    private fun findActiveCamera(): androidx.camera.core.Camera? {
        // Access VisionCamera's CameraView → cameraSession → camera via reflection.
        // This avoids pulling in camera-lifecycle just for ProcessCameraProvider.
        return try {
            val activity = reactApplicationContext.currentActivity ?: return lastCamera
            val rootView = activity.window.decorView.rootView as android.view.ViewGroup
            val cameraView = findCameraView(rootView)
            if (cameraView != null) {
                val sessionField = cameraView.javaClass.getDeclaredField("cameraSession")
                sessionField.isAccessible = true
                val session = sessionField.get(cameraView) ?: return lastCamera
                val cameraField = session.javaClass.getDeclaredField("camera")
                cameraField.isAccessible = true
                val camera = cameraField.get(session) as? androidx.camera.core.Camera
                if (camera != null) lastCamera = camera
                camera ?: lastCamera
            } else {
                lastCamera
            }
        } catch (e: Exception) {
            Log.e(TAG, "findActiveCamera error: ${e.message}", e)
            lastCamera
        }
    }

    private fun findCameraView(view: android.view.View): android.view.View? {
        if (view.javaClass.name.contains("CameraView")) return view
        if (view is android.view.ViewGroup) {
            for (i in 0 until view.childCount) {
                val found = findCameraView(view.getChildAt(i))
                if (found != null) return found
            }
        }
        return null
    }

    // -- Helper: generate gamma 2.2 tonemap curve for pre-M devices ─────────

    private fun buildGamma22Curve(): TonemapCurve {
        val numPoints = 64
        val curve = FloatArray(numPoints * 2)
        for (i in 0 until numPoints) {
            val x = i.toFloat() / (numPoints - 1)
            val y = Math.pow(x.toDouble(), 1.0 / 2.2).toFloat()
            curve[i * 2] = x
            curve[i * 2 + 1] = y
        }
        return TonemapCurve(curve, curve, curve)
    }
}
