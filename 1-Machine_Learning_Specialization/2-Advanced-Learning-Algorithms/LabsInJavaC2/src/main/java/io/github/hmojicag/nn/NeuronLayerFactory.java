package io.github.hmojicag.nn;

import org.junit.Assert;

public class NeuronLayerFactory {

    public static NeuronLayer getNeuronLayer(int units, ActivationFunctionType activationFunctionType, NeuronWeight[] weights) {
        Assert.assertTrue("At least one neuron", units >= 1);
        Assert.assertEquals(units, weights.length);
        Neuron[] neurons = new Neuron[units];
        for (int i = 0; i < units; i++) {
            neurons[i] = new Neuron(weights[i].getW(), weights[i].getB(), activationFunctionType);
        }
        return new NeuronLayer(neurons);
    }

}
