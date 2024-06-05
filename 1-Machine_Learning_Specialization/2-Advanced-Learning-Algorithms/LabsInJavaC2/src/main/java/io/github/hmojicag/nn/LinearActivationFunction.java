package io.github.hmojicag.nn;

import org.junit.Assert;

public class LinearActivationFunction extends ActivationFunctionAbstract {


    /**
     * Initializes the Activation Function with its respective weights
     *
     * @param w An array of n values. The weights of x
     * @param b The bias
     */
    public LinearActivationFunction(double[] w, double b) {
        super(w, b);
    }

    /**
     * Computes the linear regression function:
     * f(x) = w * x + b
     * (Read as "the dot product" of w and x plus b
     * @param x An array of n values. The values of x
     * @return The computed value of f(x)
     */
    @Override
    public double compute(double[] x) {
        Assert.assertEquals(x.length, w.length);
        double p = 0.0;
        p += Tools.dotProduct(x, w);
        p += b;
        return p;
    }
}
