package utils;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;

public class TestUtils {
    @Test
    public void testAnchorGenerator() {
        AnchorGenerator anchorGenerator = new AnchorGenerator(32, 128, 1024);

        List<List<Double>> cellAnchors = anchorGenerator.getCellAnchors();
        assertEquals(3, cellAnchors.size());

        List<Double> firstCell = cellAnchors.get(0);
        assertEquals(4, firstCell.size());
        assertEquals(-23.0, firstCell.get(0));
        assertEquals(-11.0, firstCell.get(1));
        assertEquals(23.0, firstCell.get(2));
        assertEquals(11.0, firstCell.get(3));

        List<List<Double>> anchors = anchorGenerator.getAnchors();
        int anchorNum = 128 * 128 * 3;
        assertEquals(anchorNum, anchors.size());

        List<Double> firstAnchor = anchors.get(0);
        assertEquals(4, firstAnchor.size());
        assertEquals(-23.0, firstAnchor.get(0));
        assertEquals(-11.0, firstAnchor.get(1));
        assertEquals(23.0, firstAnchor.get(2));
        assertEquals(11.0, firstAnchor.get(3));

        List<Double> secondAnchor = anchors.get(1);
        assertEquals(4, secondAnchor.size());
        assertEquals(-16.0, secondAnchor.get(0));
        assertEquals(-16.0, secondAnchor.get(1));
        assertEquals(16.0, secondAnchor.get(2));
        assertEquals(16.0, secondAnchor.get(3));

        List<Double> lastAnchor = anchors.get(anchorNum - 1);
        assertEquals(4, lastAnchor.size());
        assertEquals(1005.0, lastAnchor.get(0));
        assertEquals(993.0, lastAnchor.get(1));
        assertEquals(1027.0, lastAnchor.get(2));
        assertEquals(1039.0, lastAnchor.get(3));
    }
}
