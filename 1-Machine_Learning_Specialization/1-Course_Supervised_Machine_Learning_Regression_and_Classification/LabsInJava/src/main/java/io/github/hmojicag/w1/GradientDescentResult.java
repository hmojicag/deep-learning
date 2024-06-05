package io.github.hmojicag.w1;

import java.util.List;

public class GradientDescentResult {
    public Double w;               // Updated value of parameter after running gradient descent
    public Double b;               // Updated value of parameter after running gradient descent
    public List<Double> histJ;     // History of cost values
    public List<Double> histW;     // History of w
    public List<Double> histB;     // History of b

    /**
     *
     * @param w         Updated value of parameter after running gradient descent
     * @param b         Updated value of parameter after running gradient descent
     * @param histJ
     * @param histW
     * @param histB
     */
    public GradientDescentResult(Double w, Double b, List<Double> histJ, List<Double> histW, List<Double> histB) {
        this.w = w;
        this.b = b;
        this.histJ = histJ;
        this.histW = histW;
        this.histB = histB;
    }
}
