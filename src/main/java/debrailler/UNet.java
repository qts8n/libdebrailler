package debrailler;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;


public class UNet extends BaseModule {
    private static final int DEFAULT_IMAGE_SIZE = 640;

    public UNet(byte[] weights) {
        super(weights);
    }

    private Mat preProcess(Mat inputs) {
        // Resize to DEFAULT_IMAGE_SIZE
        Size defaultSize = new Size(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE);
        Mat prepInputs = new Mat(defaultSize, CvType.CV_8UC3);
        Imgproc.resize(inputs, prepInputs, defaultSize);

        Mat grayInputs = new Mat(inputs.size(), CvType.CV_8U);
        Imgproc.cvtColor(prepInputs, grayInputs, Imgproc.COLOR_BGR2GRAY);
        return Dnn.blobFromImage(grayInputs, 1 / 255.0, defaultSize, new Scalar(0), false, false);
    }

    private Mat postProcess(Mat outputs, Mat originalInputs) {
        int height = outputs.size(2);
        int width = outputs.size(3);
        int pixelNum =  width * height;
        outputs = outputs.reshape(0, 2 * pixelNum);
        Mat flatOutputs = new Mat(outputs.size(), CvType.CV_64F);
        outputs.convertTo(flatOutputs, CvType.CV_64F);
        MatOfDouble rawMat = new MatOfDouble(flatOutputs);
        List<Double> rawOutputs = rawMat.toList();
        Mat mask = Mat.zeros(new Size(width, height), CvType.CV_8U);
        for (int it = 0; it < pixelNum; it++) {
            double fgScore = rawOutputs.get(it);
            double bgScore = rawOutputs.get(it + pixelNum);
            int row = it / height;
            int col = it % height;
            if (fgScore <= bgScore) {
                mask.put(row, col, 255);
            }
        }

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.erode(mask, mask, kernel, new Point(-1, -1), 2);
        Imgproc.dilate(mask, mask, kernel, new Point(-1, -1), 3);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mask, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        mask = Mat.zeros(new Size(width, height), CvType.CV_8U);
        for (MatOfPoint cnt : contours) {
            MatOfPoint2f dst = new MatOfPoint2f();
            cnt.convertTo(dst, CvType.CV_32F);
            RotatedRect rect = Imgproc.minAreaRect(dst);
            MatOfPoint cntRect = new MatOfPoint();
            Point[] vertices = new Point[4];
            rect.points(vertices);
            cntRect.fromArray(vertices);
            List<MatOfPoint> boxContours = new ArrayList<>();
            boxContours.add(cntRect);
            Imgproc.drawContours(mask, boxContours, 0, new Scalar(255), -1);
        }

        Size defaultSize = new Size(Backbone.DEFAULT_IMAGE_SIZE, Backbone.DEFAULT_IMAGE_SIZE);
        Mat prepInputs = new Mat(defaultSize, CvType.CV_8UC3);
        // Resize original inputs
        Imgproc.resize(originalInputs, prepInputs, defaultSize);
        // Resize mask
        Imgproc.resize(mask, mask, defaultSize);

        Mat processed = Mat.zeros(mask.size(), CvType.CV_8UC3);
        prepInputs.copyTo(processed, mask);

        Imgcodecs.imwrite("assets/segmentor.jpg", processed); // TODO: delete me

        return processed;
    }

    @Override
    public Mat forward(Mat inputs) {
        Mat prepInputs = preProcess(inputs);
        Mat outputs = super.forward(prepInputs);
        return postProcess(outputs, inputs);
    }
}
