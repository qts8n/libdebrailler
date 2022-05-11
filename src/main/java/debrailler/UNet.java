package debrailler;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;


public class UNet extends BaseModule {
    private static final int CELL_PADDING = 3;

    public UNet(String onnx_path) {
        super(onnx_path);
    }

    private Mat preProcess(Mat inputs) {
        Mat grayInputs = new Mat(inputs.size(), CvType.CV_8U);
        Imgproc.cvtColor(inputs, grayInputs, Imgproc.COLOR_BGR2GRAY);
        return Dnn.blobFromImage(grayInputs, 1 / 255.0, new Size(), new Scalar(0), false, false);
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
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.erode(mask, mask, kernel, new Point(-1, -1), 2);
        Imgproc.dilate(mask, mask, kernel, new Point(-1, -1), 3);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mask, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        for (MatOfPoint cnt : contours) {
            Rect cntRect = Imgproc.boundingRect(cnt);
            cntRect.x -= CELL_PADDING;
            cntRect.y -= CELL_PADDING;
            cntRect.width += CELL_PADDING * 2;
            cntRect.height += CELL_PADDING * 2;
            Imgproc.rectangle(mask, cntRect, new Scalar(255), -1);
        }

        Mat processed = new Mat(mask.size(), CvType.CV_8UC3);
        originalInputs.copyTo(processed, mask);

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
