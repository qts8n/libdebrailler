package debrailler;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;

public class BrailleDetector implements Network {
    private final Backbone fpn;

    private final ClassificationHead classificationHead;

    private final RegressionHead regressionHead;

    public BrailleDetector(Backbone backbone, ClassificationHead clsHead, RegressionHead regHead) {
        fpn = backbone;
        classificationHead = clsHead;
        regressionHead = regHead;
    }

    private static Mat normalizeMeanVariance(Mat inputs) {
        Size defaultSize = new Size(1024, 1024);
        Scalar defaultMean = new Scalar(0.485 * 255, 0.456 * 255, 0.406 * 255);
        return Dnn.blobFromImage(inputs, 0.225 * 255, defaultSize, defaultMean, false, false);
    }

    @Override
    public Mat forward(Mat inputs) {
        inputs = normalizeMeanVariance(inputs);
        Mat backboneOutputs = fpn.forward(inputs);
        Mat clsOutputs = classificationHead.forward(backboneOutputs);
        Mat regOutputs = regressionHead.forward(backboneOutputs);
        return null;
    }
}
