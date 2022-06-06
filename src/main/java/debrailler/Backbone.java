package debrailler;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;

public class Backbone extends BaseModule {
    public static final int DEFAULT_IMAGE_SIZE = 1024;

    public Backbone(byte[] weights) {
        super(weights);
    }

    private static Mat preProcess(Mat inputs) {
        // Resize to DEFAULT_IMAGE_SIZE
        Size defaultSize = new Size(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE);
        // NOTE: Magic mean-variance normalization.
        //       Do not change unless you're a wizard!
        Scalar defaultMean = new Scalar(0.485 * 255, 0.456 * 255, 0.406 * 255);
        return Dnn.blobFromImage(inputs, 1 / (0.255 * 255), defaultSize, defaultMean, true, false);
    }

    @Override
    public Mat forward(Mat inputs) {
        Mat prepInputs = preProcess(inputs);
        module.setInput(prepInputs);
        return module.forward("output");
    }
}
