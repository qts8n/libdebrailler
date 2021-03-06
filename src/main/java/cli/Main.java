package cli;

import debrailler.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import utils.BrailleDictionary;
import utils.Config;
import utils.Detection;

import java.io.*;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {
    private static byte[] getAssetWeight(Config config, Config.Key key) {
        String path = config.get(key);
        InputStream in;

        try {
            in = new FileInputStream(path);
            byte[] fileBytes = new byte[in.available()];
            int res = in.read(fileBytes);
            in.close();
            if (res == -1) {
                throw new IOException();
            }
            return fileBytes;
        } catch (IOException e) {
            return null;
        }
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

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
        UNet unet = new UNet(getAssetWeight(config, Config.Key.UNET_PATH));
        Backbone backbone = new Backbone(getAssetWeight(config, Config.Key.BACKBONE_PATH));
        ClassificationHead clsHead = new ClassificationHead(getAssetWeight(config, Config.Key.CLS_HEAD_PATH));
        RegressionHead regHead = new RegressionHead(getAssetWeight(config, Config.Key.REG_HEAD_PATH));
        BrailleDetector detector = new BrailleDetector(unet, backbone, clsHead, regHead);

        List<Detection> outputs = detector.detect(image, 0.3, 2000);

        Size defaultSize = new Size(Backbone.DEFAULT_IMAGE_SIZE, Backbone.DEFAULT_IMAGE_SIZE);
        Mat outputImage = BrailleDetector.preProcess(image);
        Imgproc.resize(outputImage, outputImage, defaultSize);

        for (Detection d : outputs) {
            Rect bBox = d.getRect();
            Imgproc.rectangle(outputImage, bBox, new Scalar(0, 255, 0), 1);
        }

        Imgcodecs.imwrite("assets/result.jpg", outputImage); // TODO: delete me

        logger.info(String.valueOf(outputs.size()));
        logger.info(BrailleDictionary.translate(outputs));
    }
}
