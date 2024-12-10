package io.github.hmojicag;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import io.github.hmojicag.djl.HandWrittenDigitsZeroToNineDataSet;
import io.github.hmojicag.djl.HandWrittenOneToNineTranslator;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

/**
 * # UNQ_C2
 * # GRADED CELL: Sequential model
 *
 * The data set contains 5000 training examples of handwritten digits.
 * Each training example is a 20-pixel x 20-pixel grayscale image of the digit.
 * Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
 * The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector.
 * Each training examples becomes a single row in our data matrix X.
 * This gives us a 5000 x 400 matrix X where every row is a training example of a handwritten digit image.
 *
 * The second part of the training set is a 5000 x 1 dimensional vector y that contains labels for the training set
 * y = 0 if the image is of the digit 0, y = 4 if the image is of the digit 4 and so on.
 *
 * tf.random.set_seed(1234) # for consistent results
 * model = Sequential(
 *     [
 *         tf.keras.Input(shape=(400,)),    #specify input size
 *         Dense(25, activation = 'relu'),
 *         Dense(15, activation = 'relu'),
 *         Dense(10, activation = 'linear')
 *     ], name = "my_model"
 * )
 *
 * model.compile(
 *     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
 *     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
 * )
 *
 * history = model.fit(
 *     X,y,
 *     epochs=40
 * )
 */
public class MainC2W2Assignment {

    public static void main(String[] args) throws IOException, TranslateException {
        // This "Block" is the Neural Network
        SequentialBlock block = new SequentialBlock();

        // This is the input vector X with 400 features for each unit
        block.add(Blocks.batchFlattenBlock(400));

        // This is the first layer of the Neuron with 25 units and a ReLU activation
        block.add(Linear.builder().setUnits(25).build());
        block.add(Activation::relu);

        // This is the second layer of the Neuron with 15 units and a ReLU activation
        block.add(Linear.builder().setUnits(15).build());
        block.add(Activation::relu);

        // This is the output layer, which consist of 10 units with no activation (No Activation means just Linear activation)
        block.add(Linear.builder().setUnits(10).build());

        try(Model model = Model.newInstance("mlp", "PyTorch")) {
            model.setBlock(block);
            //model.setDataType(DataType.FLOAT64);

            HandWrittenDigitsZeroToNineDataSet trainingSet = new HandWrittenDigitsZeroToNineDataSet.Builder().setSampling(100, false).build();

            try (Trainer trainer = model.newTrainer(new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss()))) {
                trainer.setMetrics(new Metrics());
                trainer.initialize(new Shape(1, 400));// Shape of the entry array
                EasyTrain.fit(trainer, 40, trainingSet, trainingSet);
            }

            // Predict using the NN Model
            try (Predictor<float[], Integer> predictor = model.newPredictor(new HandWrittenOneToNineTranslator())) {
                // Compare against training set
                List<float[]> X = trainingSet.getX();
                List<float[]> Y = trainingSet.getY();

                for (int i=0; i < X.size(); i++) {
                    float[] x = X.get(i);
                    float[] y = Y.get(i);
                    Integer predicted = predictor.predict(x);
                    Integer realValue = Math.toIntExact(Math.round(y[0]));
                    //persistImageIfNotExist(i, x, predicted, realValue);
                    System.out.print(String.format("predicted=%d, realValue=%d ", predicted, realValue));
                    if (predicted == realValue) {
                        System.out.println("CORRECT");
                    } else {
                        System.out.println(String.format("MISCLASSIFIED i [%d]", i));
                    }
                }
            }

        }

    }

    private static void persistImageIfNotExist(int index, float[] x, int yP, int yR) throws IOException {
        String fileName = String.format("src/main/resources/trainingsetimgw2/x_%d_%dP_%dR.png", index, yP, yR);
        if(!Files.exists(Paths.get(fileName))) {
            BufferedImage img = new BufferedImage(20, 20, BufferedImage.TYPE_BYTE_INDEXED);
            int pixelIndex = 0;
            for(int i = 0; i < 20; i++) {
                for(int j = 0; j < 20; j++) {
                    float intensity = x[pixelIndex] * 255;// 0.0 to 1.0 range scaled to 0 to 255 range
                    int intensityByte =  Math.round(intensity);
                    img.setRGB(i, j, intensityByte);
                    pixelIndex++;
                }
            }
            File f = new File(fileName);
            ImageIO.write(img, "png", f);
        }
    }

}
