package utils;

import java.io.IOException;
import java.util.Properties;

public class Config {
    public static final String INSTANCE = "config.properties";

    private final Properties config;

    public Config(String path) {
        config = new Properties();
        try {
            config.load(getClass().getClassLoader().getResourceAsStream(path));
        } catch (NullPointerException | IOException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    public String get(Key key) {
        return config.getProperty(key.toString());
    }

    public String get(Key key, String defaultValue) {
        return config.getProperty(key.toString(), defaultValue);
    }

    public enum Key {
        BACKBONE_PATH("backbonePath"),
        CLS_HEAD_PATH("clsHeadPath"),
        REG_HEAD_PATH("regHeadPath");

        private final String name;

        Key(String name) { this.name = name; }

        public String toString() { return name; }
    }
}
