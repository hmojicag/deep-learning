package io.github.hmojicag.w2;

public class GradientResultMultiVar {
    public double[] dj_dw;
    public double dj_db;

    public GradientResultMultiVar(double[] dj_dw, double dj_db) {
        this.dj_dw = dj_dw;
        this.dj_db = dj_db;
    }
}
