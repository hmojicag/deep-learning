package io.github.hmojicag.w1;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class GradientDescentF implements Function<GradientDescentParams, GradientDescentResult> {

    /**
     * Performs gradient descent to fit w,b. Updates w,b by taking num_iters gradient steps with learning rate alpha
     */
    @Override
    public GradientDescentResult apply(GradientDescentParams params) {
        List<Double> histJ = new ArrayList<>();
        List<Double> histW = new ArrayList<>();
        List<Double> histB = new ArrayList<>();

        Double b = params.bInit;
        Double w = params.wInit;
        Double alpha = params.alpha;
        for(int i = 0; i < params.numIter; i++) {
            // Calculate the gradient and update the parameters using gradient_function
            GradientResult gradientResult = params.gradientFunction.apply(new LinearRegressionParams(params.xValues, params.yValues, w, b));
            Double dj_db = gradientResult.dj_db;
            Double dj_dw = gradientResult.dj_dw;

            // Update Parameters using equations for gradient descent:
            //      w = w - alpha dJ(w,b)       b = b - alpha dJ(w,b)
            //                      dw                          db
            b = b - alpha * dj_db;
            w = w - alpha * dj_dw;

            // Persist historic data
            Double cost = params.costFunction.apply(new LinearRegressionParams(params.xValues, params.yValues, w, b));
            histJ.add(cost);
            histW.add(w);
            histB.add(b);

            // Print cost every at intervals 10 times or as many iterations if < 10
            if(i % Math.ceil(params.numIter / 10) == 0) {
                System.out.print(String.format("Iteration %d: Cost %.4f", i, cost));
                System.out.print(String.format("dj_dw %.4f, dj_db %.4f", dj_dw, dj_db));
                System.out.println(String.format("w %.4f, b %.4f", w, b));
            }
        }

        return new GradientDescentResult(w, b, histJ, histW, histB);
    }
}
