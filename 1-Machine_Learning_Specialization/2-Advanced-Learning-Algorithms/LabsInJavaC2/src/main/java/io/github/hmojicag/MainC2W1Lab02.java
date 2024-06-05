package io.github.hmojicag;

import io.github.hmojicag.nn.ActivationFunctionType;
import io.github.hmojicag.nn.NeuralNetworkModel;
import io.github.hmojicag.nn.NeuronLayer;
import io.github.hmojicag.nn.NeuronLayerFactory;
import io.github.hmojicag.nn.NeuronWeight;
import io.github.hmojicag.nn.SequentialNeuralNetworkModel;
import org.junit.Assert;

/**
 * C2_W1_Lab02_CoffeeRoasting_TF
 *
 * Neuron with 2 layers
 * Layer 1 has 3 units
 * Layer 2 has 1 unit, this is the output layer
 *
 *  x     O  a1          a2
 * --->   O -----> O  ------>
 *        O
 *
 *      layer1   layer2
 *
 * Y = a2 >= 0.5
 * X = (x1     , x2)
 *     duration, temperature
 *
 * layer1:
 *  weights:
 *  w1 = [w1.1, w1.2]   b1 = [b1]
 *  w2 = [w2.1, w2.2]   b2 = [b2]
 *  w3 = [w3.1, w3.2]   b3 = [b3]
 *
 *  a1.1 = sigmoid( w1.1 * x1 + w1.2 * x2 + b1 )
 *  a1.2 = sigmoid( w2.1 * x1 + w2.2 * x2 + b2 )
 *  a1.3 = sigmoid( w3.1 * x1 + w3.2 * x2 + b3 )
 *
 * layer2:
 *  weights:
 *  w = [w1, w2, w3]   b = [b]
 *
 *  a2 = sigmoid( w1 * a1.1 + w2 * a1.2 + w3 * a1.3 + b )
 */
public class MainC2W1Lab02 {
    public static void main(String[] args) {
        NeuronWeight lw1 = new NeuronWeight(new double[]{-8.94, -0.17}, -9.87);
        NeuronWeight lw2 = new NeuronWeight(new double[]{0.29,  -7.34}, -9.28);
        NeuronWeight lw3 = new NeuronWeight(new double[]{12.89, 10.79}, 1.01);
        NeuronLayer logisticLayer1 = NeuronLayerFactory.getNeuronLayer(3, ActivationFunctionType.SIGMOID, new NeuronWeight[]{lw1, lw2, lw3});

        NeuronWeight lwo = new NeuronWeight(new double[]{-31.38, -27.86, -32.79}, 15.54);
        NeuronLayer logisticLayer2 = NeuronLayerFactory.getNeuronLayer(1, ActivationFunctionType.SIGMOID, new NeuronWeight[]{lwo});

        NeuralNetworkModel model = new SequentialNeuralNetworkModel(new NeuronLayer[] {logisticLayer1, logisticLayer2});
        double[] p1 = model.predict(new double[]{-0.47,0.42});    // Positive Example
        double[] p2 = model.predict(new double[]{-0.47,3.16});    // Negative Example

        Assert.assertTrue(p1[0] > 0.5);
        Assert.assertTrue(p2[0] <= 0.5);
    }
}