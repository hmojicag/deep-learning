package io.github.hmojicag;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.loss.Loss;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import io.github.hmojicag.djl.HandWrittenOneOrZeroDataSet;
import io.github.hmojicag.djl.HandWrittenOneOrZeroTranslator;
import org.slf4j.impl.SimpleLogger;

import java.util.List;

/**
 *
 * https://docs.djl.ai/
 * https://docs.djl.ai/docs/demos/jupyter/tutorial/01_create_your_first_network.html#step-2-determine-your-input-and-output-size
 * https://docs.djl.ai/examples/docs/train_mnist_mlp.html
 * https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/training/TrainMnist.java
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
 * The activations for all units are Sigmoid
 */
public class MainC2W1AssignmentDjl {


    public static void main(String[] args) throws Exception {
        // This "Block" is the Neural Network
        SequentialBlock block = new SequentialBlock();

        // This is the input vector X with 400 features for each unit
        block.add(Blocks.batchFlattenBlock(400));

        // This is the first layer of the Neuron with 25 units and a Sigmoid activation
        block.add(Linear.builder().setUnits(25).build());
        block.add(Activation::sigmoid);

        // This is the second layer of the Neuron with 15 units and a Sigmoid activation
        block.add(Linear.builder().setUnits(15).build());
        block.add(Activation::sigmoid);

        // This is the output layer, which consist of one single unit with a Sigmoid activation
        block.add(Linear.builder().setUnits(1).build());
        block.add(Activation::sigmoid);

        // Create a Neural Network model with a name "mlp" (Multilayer Perceptron)
        // Multilayer Perceptron is a nother name for a NN that connects everything to everything (Dense in Keras Tensorflow)
        // https://djl.ai/docs/engine.html
        //String engineName = new PtEngineProvider().getEngineName();
        String engineName = "PyTorch";
        try(Model model = Model.newInstance("mlp", engineName)) {
            model.setBlock(block);

            // Get training datasets
            HandWrittenOneOrZeroDataSet trainingSet = new HandWrittenOneOrZeroDataSet.Builder().setSampling(100, false).build();

            // Train the NN Model
            try (Trainer trainer = model.newTrainer(new DefaultTrainingConfig(Loss.sigmoidBinaryCrossEntropyLoss()))) {
                trainer.setMetrics(new Metrics());
                trainer.initialize(new Shape(1, 400));// Shape of the array
                EasyTrain.fit(trainer, 20, trainingSet, trainingSet);
            }

            // Predict using the NN Model
            try (Predictor<float[], float[]> predictor = model.newPredictor(new HandWrittenOneOrZeroTranslator())) {
                // Compare against training set
                List<float[]> X = trainingSet.getX();
                List<float[]> Y = trainingSet.getY();

                for (int i=0; i < X.size(); i++) {
                    float[] x = X.get(i);
                    float[] y = Y.get(i);
                    float p = predictor.predict(x)[0];
                    float p1 = p >= 0.5 ? 1 : 0;
                    System.out.print(String.format("p=%.6f, p1=%.1f, y=%.1f ", p, p1, y[0]));
                    if (p1 == y[0]) {
                        System.out.println("CORRECT");
                    } else {
                        System.out.println(String.format("MISCLASSIFIED i [%d]", i));
                    }
                }
            }
        }
    }

}
