package debrailler;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import utils.AnchorGenerator;
import utils.Detection;

import java.util.ArrayList;
import java.util.List;

public class BrailleDetector {
    private final AnchorGenerator anchorGenerator;
    
    private final UNet prep;

    private final Backbone fpn;

    private final ClassificationHead classificationHead;

    private final RegressionHead regressionHead;

    public BrailleDetector(UNet unet, Backbone backbone, ClassificationHead clsHead, RegressionHead regHead) {
        anchorGenerator = new AnchorGenerator(32, 128, Backbone.DEFAULT_IMAGE_SIZE);
        prep = unet;
        fpn = backbone;
        classificationHead = clsHead;
        regressionHead = regHead;
    }

    public static Mat preProcess(Mat inputs) {
        // Padding to square
        int width = inputs.width();
        int height = inputs.height();
        int flag = height - width;
        int top = 0;
        int bottom = 0;
        int left = 0;
        int right = 0;
        int maxSide = width;
        if (flag != 0) {
            int diff = Math.abs(flag);
            int border1 = diff / 2;
            int border2 = Math.round(diff / 2.0f);
            if (flag > 0) {
                left = border1;
                right = border2;
                maxSide = height;
            } else {
                top = border1;
                bottom = border2;
            }
        }
        Size maxSize = new Size(maxSide, maxSide);
        Mat paddedInputs = new Mat(maxSize, CvType.CV_8UC3);
        Core.copyMakeBorder(inputs, paddedInputs, top, bottom, left, right, Core.BORDER_CONSTANT, new Scalar(0));
        return paddedInputs;
    }

    private Rect2d decodeBBox(List<Double> boxReg, List<Double> anchor) {
        assert boxReg.size() == 4;
        assert anchor.size() == 4;

        // NOTE: Maybe we should compute these while generating
        //       anchors beforehand
        double width = anchor.get(2) - anchor.get(0);
        double height = anchor.get(3) - anchor.get(1);
        double cX = anchor.get(0) + 0.5 * width;
        double cY = anchor.get(1) + 0.5 * height;

        double dX = boxReg.get(0);
        double dY = boxReg.get(1);
        double dW = boxReg.get(2);
        double dH = boxReg.get(3);

        double predCX = dX * width + cX;
        double predCY = dY * height + cY;
        double predW = Math.exp(dW) * width;
        double predH = Math.exp(dH) * height;

        return new Rect2d(predCX, predCY, predW, predH);
    }

    private List<Detection> postProcess(MatOfDouble clsOutputs, MatOfDouble regOutputs, double scoreThresh, int topK) {
        int possibleDetectionNum = clsOutputs.size(0);
        assert possibleDetectionNum == regOutputs.size(0);

        List<Double> logIts = clsOutputs.toList();
        List<Double> boxRegs = regOutputs.toList();
        List<List<Double>> anchors = anchorGenerator.getAnchors();

        List<Detection> detections = new ArrayList<>();
        List<Rect2d> boxes = new ArrayList<>();
        List<Float> scores = new ArrayList<>();
        int currIt = -1;
        for (double logIt : logIts) {
            currIt++;
            // NOTE: There might not be a need to compute sigmoid
            //       from each passing score. It would be better to
            //       skip non-max scores.
            double score = 1.0 / (1.0 + Math.exp(-logIt));
            if (score < scoreThresh) {
                continue;
            }
            int label = currIt % ClassificationHead.NUM_CLASSES;
            int anchorIt = currIt / ClassificationHead.NUM_CLASSES;
            int boxRegIt = anchorIt * RegressionHead.NUM_COORDS;
            int boxRegIt1 = boxRegIt + RegressionHead.NUM_COORDS;
            List<Double> boxReg = boxRegs.subList(boxRegIt, boxRegIt1);
            List<Double> anchor = anchors.get(anchorIt);
            Rect2d box = decodeBBox(boxReg, anchor);

            boxes.add(box);
            scores.add((float) score);
            detections.add(new Detection(box, label, score));
        }

        MatOfRect2d boxMat = new MatOfRect2d();
        boxMat.fromList(boxes);
        MatOfFloat scoreMat = new MatOfFloat();
        scoreMat.fromList(scores);
        MatOfInt indexMat = new MatOfInt();
        Dnn.NMSBoxes(boxMat, scoreMat, (float) scoreThresh, 0.2f, indexMat, 1.0f, topK);

        List<Detection> filteredDetections = new ArrayList<>();
        if (indexMat.empty()) {
            return filteredDetections;
        }

        List<Integer> indices = indexMat.toList();
        for (int idx : indices) {
            filteredDetections.add(detections.get(idx));
        }
        return filteredDetections;
    }

    public List<Detection> detect(Mat inputs, double scoreThresh, int topK) {
        Mat prepInputs = preProcess(inputs);
        Mat unetOutputs = prep.forward(prepInputs);
        Mat backboneOutputs = fpn.forward(unetOutputs);
        Mat clsOutputs = classificationHead.forward(backboneOutputs);
        Mat regOutputs = regressionHead.forward(backboneOutputs);
        return postProcess(
                new MatOfDouble(clsOutputs),
                new MatOfDouble(regOutputs),
                scoreThresh,
                topK
        );
    }
}
