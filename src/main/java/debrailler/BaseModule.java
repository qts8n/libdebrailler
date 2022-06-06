package debrailler;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

abstract class BaseModule implements NetworkModule {
    protected Net module;

    BaseModule(byte[] weights) {
        MatOfByte weightMat = new MatOfByte(weights);
        module = Dnn.readNetFromONNX(weightMat);
    }

    public Mat forward(Mat inputs) {
        module.setInput(inputs);
        return module.forward();
    }
}
