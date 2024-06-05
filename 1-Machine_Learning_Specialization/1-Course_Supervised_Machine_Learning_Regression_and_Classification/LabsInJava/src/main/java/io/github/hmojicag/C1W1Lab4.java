package io.github.hmojicag;

import io.github.hmojicag.w1.GradientDescentF;
import io.github.hmojicag.w1.GradientDescentParams;
import io.github.hmojicag.w1.GradientDescentResult;
import io.github.hmojicag.w1.GradientF;
import io.github.hmojicag.w1.TotalCostJF;
import java.util.Arrays;
import java.util.List;

public class C1W1Lab4 {

    public static void main(String[] args) {
        // Data set
        List<Double> xValues = Arrays.asList(1.0, 2.0);
        List<Double> yValues = Arrays.asList(300.0, 500.0);

        // Setup Parameters for Gradient Descent
        Double wInit = 0.0;
        Double bInit = 0.0;
        int numIterations = 10000;
        Double alpha = 0.01;
        TotalCostJF costFunction = new TotalCostJF();
        GradientF gradientFunction = new GradientF();
        GradientDescentParams gradientDescentParams = new GradientDescentParams(xValues, yValues, wInit, bInit, alpha, numIterations, costFunction, gradientFunction);

        // Compute gradient descent
        GradientDescentF gradientDescentFunction = new GradientDescentF();
        GradientDescentResult gdResult = gradientDescentFunction.apply(gradientDescentParams);

        // Print results
        System.out.println(String.format("(w,b) found by gradient descent: (%.4f, %.4f)", gdResult.w, gdResult.b));
    }

}
