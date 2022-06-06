package utils;

import org.opencv.core.Rect;
import org.opencv.core.Rect2d;

public class Detection {
    public final Rect2d bBox;

    public final int label;

    public final double score;

    public Detection(Rect2d bBoxRect, int objLabel, double objScore) {
        bBox = bBoxRect;
        label = objLabel;
        score = objScore;
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
}
