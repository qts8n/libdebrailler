package debrailler;

import org.opencv.core.Mat;

public interface Network {
    public Mat forward(Mat inputs);
}
