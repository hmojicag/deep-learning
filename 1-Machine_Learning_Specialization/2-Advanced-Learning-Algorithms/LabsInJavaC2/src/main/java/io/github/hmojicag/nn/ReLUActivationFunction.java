package io.github.hmojicag.nn;

import org.junit.Assert;

public class ReLUActivationFunction extends ActivationFunctionAbstract {


    /**
     * Initializes the Activation Function with its respective weights
     *
     * @param w An array of n values. The weights of x
     * @param b The bias
     */
    public ReLUActivationFunction(double[] w, double b) {
        super(w, b);
    }

    /**
     * Computes the ReLU function:
     * g(z) = max(0,z)
     * Where z is the function f(x) for linear regression.
     * Therefore g(z) = g(f(x))
     * @param x An array of n values. The values of x
     * @return The computed value of g(f(x))
     */
    @Override
    public double compute(double[] x) {
        Assert.assertEquals(x.length, w.length);
        LinearActivationFunction linearActivationFunction = new LinearActivationFunction(w, b);
        double z = linearActivationFunction.compute(x);
        return Math.max(0, z);
    }
}
