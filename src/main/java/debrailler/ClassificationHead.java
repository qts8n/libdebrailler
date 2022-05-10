package debrailler;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class ClassificationHead extends BaseModule {
    public static final int NUM_CLASSES = 64;

    public ClassificationHead(String onnx_path) {
        super(onnx_path);
    }

    private Mat postProcess(Mat outputs) {
        int channels = outputs.size(1);
        outputs = outputs.reshape(0, channels * NUM_CLASSES);
        Mat processed = new Mat(outputs.size(), CvType.CV_64F);
        outputs.convertTo(processed, CvType.CV_64F);
        return processed;
    }

    @Override
    public Mat forward(Mat inputs) {
        Mat outputs = super.forward(inputs);
        return postProcess(outputs);
    }
}
