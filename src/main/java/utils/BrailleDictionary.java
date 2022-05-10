package utils;

import java.util.Map;

public class BrailleDictionary {
    public static final String CAPITALIZER = "45";

    public static final String TYPO = "*";

    public static final Map<String, String> RU;

    static {
        RU = Map.ofEntries(
                Map.entry("1", "а"),
                Map.entry("12", "б"),
                Map.entry("2456", "в"),
                Map.entry("1245", "г"),
                Map.entry("145", "д"),
                Map.entry("15", "е"),
                Map.entry("16", "ё"),
                Map.entry("245", "ж"),
                Map.entry("1356", "з"),
                Map.entry("24", "и"),
                Map.entry("12346", "й"),
                Map.entry("13", "к"),
                Map.entry("123", "л"),
                Map.entry("134", "м"),
                Map.entry("1345", "н"),
                Map.entry("135", "о"),
                Map.entry("1234", "п"),
                Map.entry("1235", "р"),
                Map.entry("234", "c"),
                Map.entry("2345", "т"),
                Map.entry("136", "у"),
                Map.entry("124", "ф"),
                Map.entry("125", "х"),
                Map.entry("14", "ц"),
                Map.entry("12345", "ч"),
                Map.entry("156", "ш"),
                Map.entry("1346", "щ"),
                Map.entry("12356", "ъ"),
                Map.entry("2346", "ы"),
                Map.entry("23456", "ь"),
                Map.entry("246", "э"),
                Map.entry("1256", "ю"),
                Map.entry("1246", "я"),
                Map.entry("6", "^"),
                Map.entry("2", ","),
                Map.entry("3", ","),
                Map.entry("346", "."),
                Map.entry("356", "\""),
                Map.entry("26", "^"),
                Map.entry("34", ":"),
                Map.entry("5", "'"),
                Map.entry("36", "-"),
                Map.entry("25", "-"),
                Map.entry("2356", "-"),
                Map.entry(CAPITALIZER, "_")
        );
    }

    public String numberToLetter(int label) {
        String rawBinString = Integer.toBinaryString(label);
        int currLen = rawBinString.length();
        assert currLen <= 6;
        int padLen = 6 - currLen + 1;
        StringBuilder sb = new StringBuilder();
        for (int it = 0; it < currLen; it++) {
            if (rawBinString.charAt(it) == '1') {
                sb.append(it + padLen);
            }
        }
        return RU.getOrDefault(sb.toString(), TYPO);
    }
}