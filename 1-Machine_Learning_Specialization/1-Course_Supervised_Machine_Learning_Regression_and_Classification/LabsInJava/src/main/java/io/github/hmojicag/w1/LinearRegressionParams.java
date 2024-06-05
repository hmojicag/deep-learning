package io.github.hmojicag.w1;

import java.util.List;

public class LinearRegressionParams {
    public List<Double> xValues;
    public List<Double> yValues;
    public Double w;
    public Double b;

    /**
     * @param xValues The values of x in the training set.
     * @param yValues The values of y in the training set. Target Values
     * @param w Model parameter
     * @param b Model parameter
     */
    public LinearRegressionParams(List<Double> xValues, List<Double> yValues, Double w, Double b) {
        this.xValues = xValues;
        this.yValues = yValues;
        this.w = w;
        this.b = b;
    }
}
