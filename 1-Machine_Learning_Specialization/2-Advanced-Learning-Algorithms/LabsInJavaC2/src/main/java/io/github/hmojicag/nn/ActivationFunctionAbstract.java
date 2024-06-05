package io.github.hmojicag.nn;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class ActivationFunctionAbstract implements ActivationFunction {
    protected double[] w;
    protected double b;

    /**
     * Initializes the Activation Function with its respective weights
     * @param w An array of n values. The weights of x
     * @param b The bias
     */
    public ActivationFunctionAbstract(double[] w, double b) {
        this.w = w;
        this.b = b;
    }

    @Override
    public double compute(double[] x) {
        throw new NotImplementedException();
    }

    public double[] getW() {
        return w;
    }

    public double getB() {
        return b;
    }
}
