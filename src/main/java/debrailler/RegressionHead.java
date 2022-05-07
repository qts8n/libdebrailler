package debrailler;

import org.opencv.core.Mat;

public class RegressionHead extends BaseModule {
    public RegressionHead(String onnx_path) {
        super(onnx_path);
    }

    @Override
    public Mat forward(Mat inputs) {
        Mat outputs = super.forward(inputs);
        int channels = (int) outputs.size().width;
        return outputs.reshape(0, new int[] {channels, 4});
    }
}
