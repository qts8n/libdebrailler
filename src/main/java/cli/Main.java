package cli;

import debrailler.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import utils.Config;
import utils.Detection;

import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Logger logger = Logger.getLogger("main");
        logger.setLevel(Level.ALL);

        if (args == null || args.length < 2) {
            logger.severe("Image paths (I/O) is required as first and second arguments");
            System.exit(1);
        }

        Mat image = Imgcodecs.imread(args[0], Imgcodecs.IMREAD_COLOR);
        if (image.empty()) {
            logger.severe("Could not read image from given path");
            System.exit(1);
        }

        Config config = new Config(Config.INSTANCE);
        UNet unet = new UNet(config.get(Config.Key.UNET_PATH));
        Backbone backbone = new Backbone(config.get(Config.Key.BACKBONE_PATH));
        ClassificationHead clsHead = new ClassificationHead(config.get(Config.Key.CLS_HEAD_PATH));
        RegressionHead regHead = new RegressionHead(config.get(Config.Key.REG_HEAD_PATH));
        BrailleDetector detector = new BrailleDetector(unet, backbone, clsHead, regHead);

        List<Detection> outputs = detector.detect(image, 0.4, 2000);

        Mat outputImage = image.clone();
        for (Detection d : outputs) {
            Rect bBox = d.getRect();
            Imgproc.rectangle(outputImage, bBox, new Scalar(0, 255, 0), 1);
        }
        Imgcodecs.imwrite(args[1], outputImage);
        logger.info(String.valueOf(outputs.size()));
    }
}
