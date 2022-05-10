package debrailler;

import org.opencv.core.Mat;

interface NetworkModule {
    Mat forward(Mat inputs);
}
