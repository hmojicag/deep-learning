package io.github.hmojicag.nn;

public class SequentialNeuralNetworkModel implements NeuralNetworkModel {

    private NeuronLayer[] layers;

    public SequentialNeuralNetworkModel(NeuronLayer[] layers) {
        // TODO: Validate layers weight sizes match
        this.layers = layers;
    }

    @Override
    public double[] predict(double[] x) {
        double[] activationValues = x;
        for (int i = 0; i < layers.length; i++) {
            activationValues = layers[i].computeActivationValues(activationValues);
        }
        return activationValues;
    }
}
