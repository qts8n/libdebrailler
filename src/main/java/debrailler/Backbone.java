package debrailler;

import org.opencv.core.Mat;

public class Backbone extends BaseModule {
    public Backbone(String onnx_path) {
        super(onnx_path);
    }

    @Override
    public Mat forward(Mat inputs) {
        module.setInput(inputs);
        return module.forward("output");
    }
}
