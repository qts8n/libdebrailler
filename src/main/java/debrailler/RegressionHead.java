package debrailler;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class RegressionHead extends BaseModule {
    public static final int NUM_COORDS = 4;

    public RegressionHead(byte[] weights) {
        super(weights);
    }

    private Mat postProcess(Mat outputs) {
        int channels = outputs.size(1);
        outputs = outputs.reshape(0, channels * NUM_COORDS);
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
