package io.github.hmojicag;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class C1W1Lab2 {

    public static void main(String[] args) throws PythonExecutionException, IOException {
        mapDummyTrainValues();
        mapPredictionVsActualValues();
        accuratePrediction();
    }

    private static void mapDummyTrainValues() throws PythonExecutionException, IOException {
        // Plot data points
        List<Double> xValues = Arrays.asList(1.0, 2.0);
        List<Double> yValues = Arrays.asList(300.0, 500.0);
        Plot plt = Plot.create();
        plt.ylabel("Price (in 1000s of dollars)");
        plt.xlabel("Size (1000 sqft)");
        plt.title("Housing Prices");
        plt.plot().add(xValues, yValues);
        plt.show();
    }

    private static void mapPredictionVsActualValues() throws PythonExecutionException, IOException {
        // Plot data points
        List<Double> xValues = Arrays.asList(1.0, 2.0);
        List<Double> yValues = Arrays.asList(300.0, 500.0);

        double w = 100;
        double b = 100;
        List<Double> tmpFw = computeModelOutput(xValues, w, b);

        Plot plt = Plot.create();
        plt.ylabel("Price (in 1000s of dollars)");
        plt.xlabel("Size (1000 sqft)");
        plt.title("Housing Prices");
        plt.plot().add(xValues, yValues,"o").label("Actual Values").color("green");   // Actual Values. Scattered points
        plt.plot().add(xValues, tmpFw).label("Our Prediction").color("yellow");         // Predicted Values.
        plt.legend();// Enable labels
        plt.show();
    }

    private static void accuratePrediction() {
        // Values for w and b which make the prediction accurate
        double w = 200;
        double b = 100;

        // Predict the value of y for x=1.2
        double x_i = 1.2;

        // y represents the cost of a home of x=1200 sqft
        double y = (w * x_i) + b;
        System.out.println(String.format("Cost of home $%.1f thousand dollars", y));
    }

    private static List<Double> computeModelOutput(List<Double> xValues, double w, double b) {
        List<Double> yValues = new ArrayList<>();
        for(Double x : xValues) {
            Double y = (w * x) + b;
            yValues.add(y);
        }
        return yValues;
    }

}
