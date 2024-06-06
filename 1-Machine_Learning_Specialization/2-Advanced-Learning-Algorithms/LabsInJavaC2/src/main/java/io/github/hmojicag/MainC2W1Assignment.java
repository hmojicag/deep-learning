package io.github.hmojicag;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.hmojicag.nn.ActivationFunctionType;
import io.github.hmojicag.nn.NeuralNetworkModel;
import io.github.hmojicag.nn.NeuronLayer;
import io.github.hmojicag.nn.NeuronLayerFactory;
import io.github.hmojicag.nn.NeuronWeight;
import io.github.hmojicag.nn.SequentialNeuralNetworkModel;
import org.junit.Assert;

import java.io.File;
import java.util.List;

/**
 * C2_W1_Assignment
 *
 * NN that tries to determine if an image is a number 1 or a number 0.
 * The training is composed of 1000 images of 20x20 pixels, in gray scale.
 * Each image data comes unrolled in 400 double array
 * Each element of the array represents the intensity of each gray scale pixel.
 * The data was extracted from the numpy array .npy file and serialized as json for easier manipulation
 * Also using matplot lib I was able to create .PNG image file of each training image
 *
 * The trained weights for the NN are serialized in JSON files in a 2D Array (Or list of arrays)
 * For layer 1 (W1.json)
 * Is a list of 25 entries, one for each Neuron Unit. Each entry has 400 values.
 *
 * For layer 2 (W2.json)
 * Is a list of 15 entries, one for each Neuron Unit. Each entry has 25 values.
 *
 * For layer 3 (W3.json)
 * Is a list of 1 entries, for the one single Neuron Unit. The entry has 15 values.
 *
 * Neuron with 3 layers
 * Layer 1 has 25 units
 * Layer 2 has 15 unit
 * Layer 3 has 1  unit, this is the output layer
 *
 *        O
 *  x     O  a1         O     a2       a3
 * --->   O ----------> O  ------> O ------->
 *        O             O
 *        O
 *      25 units      15 units    1 unit
 *      layer1         layer2    layer 3
 *
 */
public class MainC2W1Assignment {
    public static void main(String[] args) throws Exception {
        // Reads the training set
        ObjectMapper mapper = new ObjectMapper();
        List<double[]> X = mapper.readValue(new File("src/main/resources/X.json"), new TypeReference<List<double[]>>(){});
        List<double[]> Y = mapper.readValue(new File("src/main/resources/y.json"), new TypeReference<List<double[]>>(){});
        List<double[]> W1 = mapper.readValue(new File("src/main/resources/W1.json"), new TypeReference<List<double[]>>(){});
        List<double[]> W2 = mapper.readValue(new File("src/main/resources/W2.json"), new TypeReference<List<double[]>>(){});
        List<double[]> W3 = mapper.readValue(new File("src/main/resources/W3.json"), new TypeReference<List<double[]>>(){});
        double[] b1 = mapper.readValue(new File("src/main/resources/b1.json"), new TypeReference<double[]>(){});
        double[] b2 = mapper.readValue(new File("src/main/resources/b2.json"), new TypeReference<double[]>(){});
        double[] b3 = mapper.readValue(new File("src/main/resources/b3.json"), new TypeReference<double[]>(){});

        /********************************************** Build the NN. *************************************************/

        // Layer 1
        NeuronWeight[] w1 = new NeuronWeight[25];
        for(int i=0; i< w1.length; i++) {
            double[] w = W1.get(i);
            double b = b1[i];
            w1[i] = new NeuronWeight(w, b);
        }
        NeuronLayer logisticLayer1 = NeuronLayerFactory.getNeuronLayer(25, ActivationFunctionType.SIGMOID, w1);

        // Layer 2
        NeuronWeight[] w2 = new NeuronWeight[15];
        for(int i=0; i< w2.length; i++) {
            double[] w = W2.get(i);
            double b = b2[i];
            w2[i] = new NeuronWeight(w, b);
        }
        NeuronLayer logisticLayer2 = NeuronLayerFactory.getNeuronLayer(15, ActivationFunctionType.SIGMOID, w2);

        // Layer 3
        NeuronWeight[] w3 = new NeuronWeight[1];
        for(int i=0; i< w3.length; i++) {
            double[] w = W3.get(i);
            double b = b3[i];
            w3[i] = new NeuronWeight(w, b);
        }
        NeuronLayer logisticLayer3 = NeuronLayerFactory.getNeuronLayer(1, ActivationFunctionType.SIGMOID, w3);

        // Build the model
        NeuralNetworkModel model = new SequentialNeuralNetworkModel(new NeuronLayer[] {logisticLayer1, logisticLayer2, logisticLayer3});

        // Compare against training set
        for(int i=0; i < X.size(); i++) {
            double[] x = X.get(i);
            double[] y = Y.get(i);
            double p = model.predict(x)[0];
            double p1 = p >= 0.5 ? 1.0 : 0.0;
            System.out.print(String.format("p=%.6f, p1=%.1f, y=%.1f ", p, p1, y[0]));
            if (p1 == y[0]) {
                System.out.println("CORRECT");
            } else {
                System.out.println("MISCLASSIFIED");
            }
        }
    }
}