package io.github.hmojicag.nn;

import org.junit.Assert;

public class Neuron {
    private double[] w;
    private double b;
    private ActivationFunctionType activationFunctionType;
    private ActivationFunction activationFunction;

    public Neuron(double[] w, double b, ActivationFunctionType activationFunctionType) {
        this.w = w;
        this.b = b;
        this.activationFunctionType = activationFunctionType;
        activationFunction = ActivationFunctionFactory.getActivationFunction(activationFunctionType, w, b);
    }

    /**
     * Uses the activation function passed as parameter.
     * This is it? A Neuron is just a fricking wrapper to the activation function?
     * @param x
     * @return
     */
    public double activate(double[] x) {
        Assert.assertEquals(x.length, w.length);
        return activationFunction.compute(x);
    }

}
