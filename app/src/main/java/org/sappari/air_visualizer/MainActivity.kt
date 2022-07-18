package org.sappari.air_visualizer

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.sappari.air_visualizer.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.aruco.Aruco
import org.opencv.aruco.Dictionary
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.utils.Converters
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var camera: Camera
    private var backgroundCaptureState = BackgroundCaptureState.RELEASED

    companion object {
        private const val TAG = "CameraXApp"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    init {
        OpenCVLoader.initDebug()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Hide status bar and navigation bar
        window.decorView.windowInsetsController?.hide(
            WindowInsets.Type.statusBars() or WindowInsets.Type.navigationBars()
        )

        // Request camera permissions
        if (allPermissionGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Start, Stop button
        viewBinding.buttonStart.setOnClickListener {
            when (backgroundCaptureState) {
                BackgroundCaptureState.RELEASED -> {
                    setFocusFix()
                    backgroundCaptureState = BackgroundCaptureState.CAPTURING
                    viewBinding.buttonStart.text = getString(R.string.stop)
                    viewBinding.radioLaminarFlow.isEnabled = false
                    viewBinding.radioTurbulentFlow.isEnabled = false
                    viewBinding.buttonReset.isEnabled = true
                    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                }
                else -> {
                    setFocusAuto()
                    backgroundCaptureState = BackgroundCaptureState.RELEASED
                    viewBinding.buttonStart.text = getString(R.string.start)
                    viewBinding.radioLaminarFlow.isEnabled = true
                    viewBinding.radioTurbulentFlow.isEnabled = true
                    viewBinding.buttonReset.isEnabled = false
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                }
            }
        }

        // Background reset button
        viewBinding.buttonReset.setOnClickListener {
            when (backgroundCaptureState) {
                BackgroundCaptureState.CAPTURED -> {
                    backgroundCaptureState = BackgroundCaptureState.CAPTURING
                }
                else -> {}
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // ImageAnalyzer
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(1920, 1080))
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ImageAnalyzer())
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()
                // Bind use cases to camera
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    private fun setFocusFix() {
        val autoFocusPoint = SurfaceOrientedMeteringPointFactory(1f, 1f).createPoint(0.5f, 0.5f)
        val autoFocusAction = FocusMeteringAction.Builder(
            autoFocusPoint,
            FocusMeteringAction.FLAG_AF
        ).apply {
            disableAutoCancel()
        }.build()
        camera.cameraControl.startFocusAndMetering(autoFocusAction)
    }

    private fun setFocusAuto() {
        val autoFocusPoint = SurfaceOrientedMeteringPointFactory(1f, 1f).createPoint(0.5f, 0.5f)
        val autoFocusAction = FocusMeteringAction.Builder(
            autoFocusPoint,
            FocusMeteringAction.FLAG_AF
        ).apply {
            setAutoCancelDuration(5, TimeUnit.SECONDS)
        }.build()
        camera.cameraControl.startFocusAndMetering(autoFocusAction)
    }

    // ---------------------------------------------------------------------------------------------
    // ImageAnalyzer
    //
    private inner class ImageAnalyzer: ImageAnalysis.Analyzer {
        private lateinit var background: Mat
        private var markerDetectionLoopCount = 0
        private var backgroundCaptureDelayCount = 0
        private val recentQuadMarkersQueue = ArrayDeque<QuadMarkers>()

        // ArUco
        private val dictionary: Dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_4X4_50)

        // FPS
        private var previousTime = 0L
        private var fpsLoopCount = 0
        private var lastFps: String = ""

        override fun analyze(image: ImageProxy) {
            if (viewBinding.radioTurbulentFlow.isChecked) {
                analyzeTurbulentFlow(image)
            } else if (viewBinding.radioLaminarFlow.isChecked) {
                analyzeLaminarFlow(image)
            }
        }

        private fun analyzeLaminarFlow(image: ImageProxy) {
            val originalFrame = getMatFromImage(image)
            val gray = convertColorToGray(originalFrame)
            when (backgroundCaptureState) {
                BackgroundCaptureState.CAPTURING -> {
                    val quadMarkers = detectQuadMarkers(gray)
                    if (quadMarkers != null) {
                        addToRecentQuadMarkers(quadMarkers)
                        backgroundCaptureDelayCount++
                        if (backgroundCaptureDelayCount >= 10) {
                            // Slight delay to prevent blurring when tapping the Start button.
                            background = extractMarkerArea(gray, averageRecentQuadMarkers())
                            backgroundCaptureDelayCount = 0
                            backgroundCaptureState = BackgroundCaptureState.CAPTURED
                        }
                    }
                    if (recentQuadMarkersQueue.isEmpty()) {
                        showImage(gray, image)
                    } else {
                        val markerArea = extractMarkerArea(gray, averageRecentQuadMarkers())
                        showImage(markerArea, image)
                    }
                }
                BackgroundCaptureState.RELEASED -> {
                    val quadMarkers = detectQuadMarkers(gray)
                    if (quadMarkers != null) {
                        addToRecentQuadMarkers(quadMarkers)
                        drawMarkerArea(originalFrame, quadMarkers)
                    }
                    showImage(originalFrame, image)
                }
                BackgroundCaptureState.CAPTURED -> {
                    // Update quadMarkers
                    markerDetectionLoopCount++
                    if (markerDetectionLoopCount >= 20) {
                        // Marker detection is slow, so run at intervals
                        val quadMarkers = detectQuadMarkers(gray)
                        if (quadMarkers != null) {
                            addToRecentQuadMarkers(quadMarkers)
                        }
                        markerDetectionLoopCount = 0
                    }
                    // Image Processing
                    val averageQuadMarkers = averageRecentQuadMarkers()
                    val markerArea = extractMarkerArea(gray, averageQuadMarkers)
                    val diff = absdiff(markerArea, background)
                    val contrast = emphasizeContrast(diff)
                    // Show
                    showImage(contrast, image)
                }
            }
        }

        private fun analyzeTurbulentFlow(image: ImageProxy) {
            val originalFrame = getMatFromImage(image)
            val gray = convertColorToGray(originalFrame)
            when (backgroundCaptureState) {
                BackgroundCaptureState.CAPTURING -> {
                    background = gray
                    backgroundCaptureState = BackgroundCaptureState.CAPTURED
                    showImage(originalFrame, image)
                }
                BackgroundCaptureState.RELEASED -> {
                    showImage(originalFrame, image)
                }
                BackgroundCaptureState.CAPTURED -> {
                    val diff = absdiff(gray, background)
                    val contrast = emphasizeContrast(diff)
                    background = gray
                    showImage(contrast, image)
                }
            }
        }

        //
        // Draw marker area
        //
        private fun drawMarkerArea(frame: Mat, quadMarkers: QuadMarkers) {
            drawMarkerCircle(frame, quadMarkers.topLeftMarker.topLeft)
            drawMarkerCircle(frame, quadMarkers.topRightMarker.topRight)
            drawMarkerCircle(frame, quadMarkers.bottomRightMarker.bottomRight)
            drawMarkerCircle(frame, quadMarkers.bottomLeftMarker.bottomLeft)
            drawEdges(frame, quadMarkers)
        }

        private fun drawMarkerCircle(frame: Mat, point: Point) {
            val color = Scalar(0.0, 0.0, 255.0)
            Imgproc.circle(frame, point, 12, color, -1)
        }

        private fun drawEdges(frame: Mat, quadMarkers: QuadMarkers): Mat {
            val points = listOf(
                MatOfPoint(
                    quadMarkers.topLeftMarker.topLeft,
                    quadMarkers.topRightMarker.topRight,
                    quadMarkers.bottomRightMarker.bottomRight,
                    quadMarkers.bottomLeftMarker.bottomLeft
                )
            )
            val color = Scalar(0.0, 0.0, 255.0)
            Imgproc.polylines(frame, points, true, color, 2)
            return frame
        }

        //
        // Add to recentQuadMarkersQueue and average
        //
        private fun addToRecentQuadMarkers(quadMarkers: QuadMarkers) {
            recentQuadMarkersQueue.add(quadMarkers)
            if (recentQuadMarkersQueue.size > 5) {
                recentQuadMarkersQueue.removeFirst()
            }
        }

        private fun averageRecentQuadMarkers(): QuadMarkers {
            val size = recentQuadMarkersQueue.size.toDouble()
            val p = Point(0.0, 0.0)
            var averageTopLeftMarker = Marker(p, p, p, p)
            var averageTopRightMarker = Marker(p, p, p, p)
            var averageBottomRightMarker = Marker(p, p, p, p)
            var averageBottomLeftMarker = Marker(p, p, p, p)
            for (quadMarkers in recentQuadMarkersQueue) {
                averageTopLeftMarker = averageTopLeftMarker.plus(quadMarkers.topLeftMarker.divide(size))
                averageTopRightMarker = averageTopRightMarker.plus(quadMarkers.topRightMarker.divide(size))
                averageBottomRightMarker = averageBottomRightMarker.plus(quadMarkers.bottomRightMarker.divide(size))
                averageBottomLeftMarker = averageBottomLeftMarker.plus(quadMarkers.bottomLeftMarker.divide(size))
            }
            return QuadMarkers(
                averageTopLeftMarker,
                averageTopRightMarker,
                averageBottomRightMarker,
                averageBottomLeftMarker
            )
        }

        //
        // Extract the area enclosed by the four markers
        //
        private fun extractMarkerArea(frame: Mat, quadMarkers: QuadMarkers): Mat {
            val src = Converters.vector_Point_to_Mat(
                listOf(
                    quadMarkers.topLeftMarker.topLeft,
                    quadMarkers.topRightMarker.topRight,
                    quadMarkers.bottomRightMarker.bottomRight,
                    quadMarkers.bottomLeftMarker.bottomLeft
                ), CvType.CV_32F
            )
            val width = frame.width().toDouble()
            val height = frame.height().toDouble()
            val dest = Converters.vector_Point_to_Mat(
                listOf(
                    Point(0.0, 0.0),
                    Point(width, 0.0),
                    Point(width, height),
                    Point(0.0, height)
                ), CvType.CV_32F
            )
            val mat = Imgproc.getPerspectiveTransform(src, dest)
            val result = Mat(frame.size(), frame.type())
            Imgproc.warpPerspective(frame, result, mat, Size(width, height))
            return result
        }

        private fun detectQuadMarkers(gray: Mat): QuadMarkers? {
            val corners = ArrayList<Mat>()
            val ids = Mat()
            Aruco.detectMarkers(gray, dictionary, corners, ids)
            var topLeftMarker: Marker? = null
            var topRightMarker: Marker? = null
            var bottomRightMarker: Marker? = null
            var bottomLeftMarker: Marker? = null
            for (markerIndex in 0 until ids.rows()) {
                val data = IntArray(1)
                ids.get(markerIndex, 0, data)
                val id = data[0]
                val markerCorners = corners[markerIndex]
                val marker = createMarkerFromCorners(markerCorners)
                when (id) {
                    0 -> topLeftMarker = marker
                    1 -> topRightMarker = marker
                    2 -> bottomRightMarker = marker
                    3 -> bottomLeftMarker = marker
                }
            }
            if (topLeftMarker == null || topRightMarker == null
                || bottomRightMarker == null || bottomLeftMarker == null) {
                return null
            }
            return QuadMarkers(
                topLeftMarker,
                topRightMarker,
                bottomRightMarker,
                bottomLeftMarker
            )
        }

        private fun createMarkerFromCorners(markerCorners: Mat): Marker {
            var topLeft = Point()
            var topRight = Point()
            var bottomRight = Point()
            var bottomLeft = Point()
            for (cornerIndex in 0 until markerCorners.cols()) {
                val corner = FloatArray(2)
                markerCorners.get(0, cornerIndex, corner)
                val x = corner[0]
                val y = corner[1]
                val point = Point(x.toDouble(), y.toDouble())
                when (cornerIndex) {
                    0 -> topLeft = point
                    1 -> topRight = point
                    2 -> bottomRight = point
                    3 -> bottomLeft = point
                }
            }
            return Marker(topLeft, topRight, bottomRight, bottomLeft)
        }

        //
        // absdiff
        //
        private fun absdiff(src1: Mat, src2: Mat): Mat {
            val dst = Mat()
            Core.absdiff(src1, src2, dst)
            return dst
        }

        //
        // emphasizeContrast
        //
        private fun emphasizeContrast(src: Mat): Mat {
            val dst = Mat()
            //Imgproc.threshold(src, dst, 240.0, 255.0, Imgproc.THRESH_TOZERO_INV)
            //Core.multiply(dst, Scalar(3.0), dst)
            Core.multiply(src, Scalar(6.0), dst)
            return dst
        }

        //
        // showImage
        //
        private fun showImage(frame: Mat, image: ImageProxy) {
            drawFps(frame)
            val bitmap = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(frame, bitmap)
            runOnUiThread {
                viewBinding.imageView.setImageBitmap(bitmap)
            }
            image.close()
        }

        //
        // FPS
        //
        private fun drawFps(frame: Mat) {
            val fps = getFps()
            Imgproc.putText(
                frame,
                fps,
                Point(10.0, 100.0),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                1.0,
                Scalar(255.0, 255.0, 255.0)
            )
        }

        private fun getFps(): String {
            val currentTime = System.currentTimeMillis()
            if (previousTime == 0L) {
                previousTime = currentTime
            }
            if (fpsLoopCount == 20) {
                val fps = 20.0 / ((currentTime - previousTime) / 1000.0)
                lastFps = "%4.1f".format(fps)
                previousTime = currentTime
                fpsLoopCount = 0
            }
            fpsLoopCount++
            return lastFps
        }

        //
        // convertColorToGray
        //
        private fun convertColorToGray(src: Mat): Mat {
            val dst = Mat()
            Imgproc.cvtColor(src, dst, Imgproc.COLOR_RGBA2GRAY)
            return dst
        }

        //
        // getMatFromImage
        //
        private fun getMatFromImage(image: ImageProxy): Mat {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer
            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()
            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)
            val yuvMat = Mat(
                image.height + image.height / 2,
                image.width,
                CvType.CV_8UC1
            )
            yuvMat.put(0, 0, nv21)
            val rgbaMat = Mat()
            Imgproc.cvtColor(yuvMat, rgbaMat, Imgproc.COLOR_YUV2RGBA_NV21, 4)
            return rgbaMat
        }
    }

    private enum class BackgroundCaptureState {
        RELEASED,
        CAPTURING,
        CAPTURED,
    }

    private inner class Marker(
        val topLeft: Point,
        val topRight: Point,
        val bottomRight: Point,
        val bottomLeft: Point
        ) {
        fun divide(value: Double): Marker {
            return Marker(
                Point(
                    topLeft.x / value,
                    topLeft.y / value
                ),
                Point(
                    topRight.x / value,
                    topRight.y / value
                ),
                Point(
                    bottomRight.x / value,
                    bottomRight.y / value
                ),
                Point(
                    bottomLeft.x / value,
                    bottomLeft.y / value
                )
            )
        }

        fun plus(marker: Marker): Marker {
            return Marker(
                Point(
                    topLeft.x + marker.topLeft.x,
                    topLeft.y + marker.topLeft.y
                ),
                Point(
                    topRight.x + marker.topRight.x,
                    topRight.y + marker.topRight.y
                ),
                Point(
                    bottomRight.x + marker.bottomRight.x,
                    bottomRight.y + marker.bottomRight.y
                ),
                Point(
                    bottomLeft.x + marker.bottomLeft.x,
                    bottomLeft.y + marker.bottomRight.y
                )
            )
        }
    }

    private inner class QuadMarkers(
        val topLeftMarker: Marker,
        val topRightMarker: Marker,
        val bottomRightMarker: Marker,
        val bottomLeftMarker: Marker
    )
}