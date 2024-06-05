package io.github.hmojicag.nn;

import org.junit.Assert;

public class SigmoidActivationFunction extends ActivationFunctionAbstract {

    /**
     * Initializes the Activation Function with its respective weights
     * @param w An array of n values. The weights of x
     * @param b The bias
     */
    public SigmoidActivationFunction(double[] w, double b) {
        super(w, b);
    }

    /**
     * Computes the sigmoid function for logistic regression.
     * g(z) = 1 / (1 + e^-z)
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
        return 1.0 / (1.0 + Math.exp(-1*z));
    }
}
