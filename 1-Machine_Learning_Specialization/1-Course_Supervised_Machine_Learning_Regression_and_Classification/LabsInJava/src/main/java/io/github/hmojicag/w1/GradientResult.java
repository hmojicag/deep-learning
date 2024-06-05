package io.github.hmojicag.w1;

public class GradientResult {
    public Double dj_dw;
    public Double dj_db;

    public GradientResult(Double dj_dw, Double dj_db) {
        this.dj_dw = dj_dw;
        this.dj_db = dj_db;
    }
}
