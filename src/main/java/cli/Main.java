package cli;

import debrailler.Backbone;
import debrailler.BrailleDetector;
import debrailler.ClassificationHead;
import debrailler.RegressionHead;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import utils.Config;

import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {
    public static void main(String[] args) {
        Logger logger = Logger.getLogger("main");
        logger.setLevel(Level.ALL);

        if (args == null || args.length < 1) {
            logger.severe("Image path is required as the first argument");
            System.exit(1);
        }

        Mat image = Imgcodecs.imread(args[0], Imgcodecs.IMREAD_COLOR);
        if (image.empty()) {
            logger.severe("Could not read image from given path");
            System.exit(1);
        }

        Config config = new Config(Config.INSTANCE);
        Backbone backbone = new Backbone(config.get(Config.Key.BACKBONE_PATH));
        ClassificationHead clsHead = new ClassificationHead(config.get(Config.Key.CLS_HEAD_PATH));
        RegressionHead regHead = new RegressionHead(config.get(Config.Key.REG_HEAD_PATH));
        BrailleDetector detector = new BrailleDetector(backbone, clsHead, regHead);

        Mat outputs = detector.forward(image);
        logger.info(outputs.size().toString());
    }
}
