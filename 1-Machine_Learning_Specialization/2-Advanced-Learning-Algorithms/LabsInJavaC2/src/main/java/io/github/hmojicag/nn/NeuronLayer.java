package io.github.hmojicag.nn;

public class NeuronLayer {
    private Neuron[] neurons;

    public NeuronLayer(Neuron[] neurons) {
        this.neurons = neurons;
    }

    public double[] computeActivationValues(double[] x) {
        double[] activationValues = new double[neurons.length];
        for(int i = 0; i < neurons.length; i++) {
            Neuron neuron = neurons[i];
            activationValues[i] = neuron.activate(x);
        }
        return activationValues;
    }

}
