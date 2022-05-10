package utils;

import org.opencv.core.Rect;
import org.opencv.core.Rect2d;

public class Detection {
    private final Rect2d bBox;

    private final int label;

    private final double score;

    public Detection(Rect2d bBoxRect, int objLabel, double objScore) {
        bBox = bBoxRect;
        label = objLabel;
        score = objScore;
    }

    public double getScore() {
        return score;
    }

    public Rect2d getBBox() {
        return bBox;
    }

    public Rect getRect() {
        double x = bBox.x - bBox.width * 0.5;
        double y = bBox.y - bBox.height * 0.5;
        return new Rect(
                (int) x,
                (int) y,
                (int) bBox.width,
                (int) bBox.height
        );
    }

    public int getLabel() {
        return label;
    }
}
