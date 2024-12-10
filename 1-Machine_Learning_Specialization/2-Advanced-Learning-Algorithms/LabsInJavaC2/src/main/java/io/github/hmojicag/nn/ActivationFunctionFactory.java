package io.github.hmojicag.nn;

public class ActivationFunctionFactory {

    /**
     * Instantiates a new type of activation function based on the parameters passed
     */
    public static ActivationFunction getActivationFunction(ActivationFunctionType type, double[] w, double b) {
        if (type == ActivationFunctionType.LINEAR) {
            return new LinearActivationFunction(w, b);
        } else if (type == ActivationFunctionType.SIGMOID) {
            return new SigmoidActivationFunction(w, b);
        } else if (type == ActivationFunctionType.RELU) {
            return new ReLUActivationFunction(w, b);
        }
        throw new IllegalArgumentException("Can't recognize ActivationFunctionType");
    }

}
