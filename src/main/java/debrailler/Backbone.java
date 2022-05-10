package debrailler;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;

public class Backbone extends BaseModule {
    public static final int DEFAULT_IMAGE_SIZE = 1024;
//    public static final int DEFAULT_IMAGE_SIZE = 768;

    public Backbone(String onnx_path) {
        super(onnx_path);
    }

    private static Mat preProcess(Mat inputs) {
        Size defaultSize = new Size(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE);
        // NOTE: Magic mean-variance normalization.
        //       Do not change unless you're a wizard!
        Scalar defaultMean = new Scalar(0.485 * 255, 0.456 * 255, 0.406 * 255);
        return Dnn.blobFromImage(inputs, 1 / (0.255 * 255), defaultSize, defaultMean, false, false);
    }

    @Override
    public Mat forward(Mat inputs) {
        Mat prepInputs = preProcess(inputs);
        module.setInput(prepInputs);
        return module.forward("output");
    }
}
