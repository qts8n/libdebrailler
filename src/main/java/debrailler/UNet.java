package debrailler;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.List;


public class UNet extends BaseModule {
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
        int pixelNum = outputs.size(1) * width * height;
        outputs = outputs.reshape(0, pixelNum);
        Mat flatOutputs = new Mat(outputs.size(), CvType.CV_64F);
        outputs.convertTo(flatOutputs, CvType.CV_64F);
        MatOfDouble rawMat = new MatOfDouble(flatOutputs);
        List<Double> rawOutputs = rawMat.toList();
        Mat mask = new Mat(new Size(width, height), CvType.CV_8U);
        for (int it = 0; it < pixelNum; it += 2) {
            double fgScore = rawOutputs.get(it);
            double bgScore = rawOutputs.get(it + 1);
            int ij = it / 2;
            int row = ij / height;
            int col = ij % height;
            if (fgScore > bgScore) {
                mask.put(row, col, 0);
            } else {
                mask.put(row, col, 255);
            }
        }
        Imgcodecs.imwrite("segmentor.jpg", mask); // TODO: delete me

        Mat processed = new Mat(new Size(width, height), CvType.CV_8UC3);
        Core.bitwise_and(originalInputs, originalInputs, processed, mask);
        return processed;
    }

    @Override
    public Mat forward(Mat inputs) {
        Mat prepInputs = preProcess(inputs);
        Mat outputs = super.forward(prepInputs);
        return postProcess(outputs, inputs);
    }
}
