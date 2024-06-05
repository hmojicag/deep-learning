package io.github.hmojicag.w1;
import java.util.List;
import java.util.function.Function;

public class GradientDescentParams {
    public List<Double> xValues;
    public List<Double> yValues;
    public Double wInit;
    public Double bInit;
    public Double alpha;
    public int numIter;
    public Function<LinearRegressionParams, Double> costFunction;
    public Function<LinearRegressionParams, GradientResult> gradientFunction;

    /**
     * @param xValues
     * @param yValues
     * @param wInit       initial values of model parameters
     * @param bInit       initial values of model parameters
     * @param alpha     Learning rate
     * @param numIter   number of iterations to run gradient descent
     * @param costFunction function that computes the cost
     * @param costFunction function that computes the gradient
     */
    public GradientDescentParams(List<Double> xValues, List<Double> yValues, Double wInit, Double bInit, Double alpha, int numIter, Function<LinearRegressionParams, Double> costFunction, Function<LinearRegressionParams, GradientResult> gradientFunction) {
        this.xValues = xValues;
        this.yValues = yValues;
        this.wInit = wInit;
        this.bInit = bInit;
        this.alpha = alpha;
        this.numIter = numIter;
        this.costFunction = costFunction;
        this.gradientFunction = gradientFunction;
    }
}
