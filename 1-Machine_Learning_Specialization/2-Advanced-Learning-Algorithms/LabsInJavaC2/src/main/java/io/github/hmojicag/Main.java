package io.github.hmojicag;

import io.github.hmojicag.nn.ActivationFunctionType;
import io.github.hmojicag.nn.NeuronLayer;
import io.github.hmojicag.nn.NeuronLayerFactory;
import io.github.hmojicag.nn.NeuronWeight;
import org.junit.Assert;

/**
 * C2_W1_Lab01_Neurons_and_Layers
 */
public class Main {
    public static void main(String[] args) {
        double[] xTrain = new double[] {1.0, 2.0};
        double[] yTrain = new double[] {300.0, 500.0};

        // w*x + b
        // 200x + 100
        NeuronWeight w1 = new NeuronWeight(new double[]{200}, 100);
        NeuronLayer linearLayer = NeuronLayerFactory.getNeuronLayer(1, ActivationFunctionType.LINEAR, new NeuronWeight[]{w1});

        double[] x1 = new double[] {xTrain[0]};
        double[] yPredicted1 = linearLayer.computeActivationValues(x1);

        double[] x2 = new double[] {xTrain[1]};
        double[] yPredicted2 = linearLayer.computeActivationValues(x2);

        Assert.assertEquals(yTrain[0], yPredicted1[0], 0.001);
        Assert.assertEquals(yTrain[1], yPredicted2[0], 0.001);


        // w*x + b
        // 200x + 100
        NeuronWeight lw1 = new NeuronWeight(new double[]{2}, -4.5);
        NeuronLayer logisticLayer = NeuronLayerFactory.getNeuronLayer(1, ActivationFunctionType.SIGMOID, new NeuronWeight[]{lw1});

        double[] lx1 = new double[] {0.0};
        double[] lyPredicted1 = logisticLayer.computeActivationValues(lx1);

        Assert.assertEquals(0.01, lyPredicted1[0], 0.001);
    }
}