package utils;

import java.util.ArrayList;
import java.util.List;

public class AnchorGenerator {
    private final List<List<Double>> cellAnchors;
    private final List<List<Double>> anchors;

    public AnchorGenerator(int anchorSize, int gridSize, int imageSize) {
        int strideSize = imageSize / gridSize;
        cellAnchors = generateCellAnchors(anchorSize);
        anchors = generateAnchors(gridSize, strideSize);
    }

    private static List<Double> computeCellAnchorForRatio(int scale, double ratio) {
        double hRatio = Math.sqrt(ratio);
        double wRatio = 1.0 / hRatio;
        hRatio *= scale;
        wRatio *= scale;
        hRatio = Math.round(hRatio / 2.0);
        wRatio = Math.round(wRatio / 2.0);
        return List.of(-wRatio, -hRatio, wRatio, hRatio);
    }

    private static List<List<Double>> generateCellAnchors(int scale) {
        return List.of(
                computeCellAnchorForRatio(scale, 0.5),
                computeCellAnchorForRatio(scale, 1.0),
                computeCellAnchorForRatio(scale, 2.0)
        );
    }

    private List<List<Double>> generateAnchors(int gridSize, int strideSize) {
        List<List<Double>> anchors = new ArrayList<>();
        for (int shiftY = 0; shiftY < gridSize; shiftY++) {
            int shiftC1 = shiftY * strideSize;
            for (int shiftX = 0; shiftX < gridSize; shiftX++) {
                int shiftC2 = shiftX * strideSize;
                for (List<Double> cell : cellAnchors) {
                    double x1 = shiftC2 + cell.get(0);
                    double y1 = shiftC1 + cell.get(1);
                    double x2 = shiftC2 + cell.get(2);
                    double y2 = shiftC1 + cell.get(3);

                    anchors.add(List.of(
                            x1,
                            y1,
                            x2,
                            y2
                    ));
                }
            }
        }
        return anchors;
    }

    public List<List<Double>> getCellAnchors() {
        return cellAnchors;
    }

    public List<List<Double>> getAnchors() {
        return anchors;
    }
}
