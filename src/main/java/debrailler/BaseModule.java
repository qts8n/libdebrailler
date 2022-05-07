package debrailler;

import org.opencv.core.Mat;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

public abstract class BaseModule implements Network {
    protected Net module;

    BaseModule(String onnx_path) {
        module = Dnn.readNetFromONNX(onnx_path);
    }

    public Mat forward(Mat inputs) {
        module.setInput(inputs);
        return module.forward();
    }
}