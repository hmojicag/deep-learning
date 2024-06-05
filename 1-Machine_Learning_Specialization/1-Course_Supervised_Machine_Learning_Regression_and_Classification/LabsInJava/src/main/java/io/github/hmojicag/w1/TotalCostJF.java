package io.github.hmojicag.w1;

import java.util.List;
import java.util.function.Function;

public class TotalCostJF implements Function<LinearRegressionParams, Double> {
    @Override
    public Double apply(LinearRegressionParams params) {
        Double cost = 0.0;
        int m = params.xValues.size();
        List<Double> xValues = params.xValues;
        List<Double> yValues = params.yValues;
        Double w = params.w;
        Double b = params.b;
        for(int i = 0; i < m; i++) {
            Double x = xValues.get(i);
            Double y = yValues.get(i);
            Double f_wb = (w * x) + b;
            cost = cost + Math.pow(f_wb - y, 2);
        }
        return 1 / (2 * m) * cost;
    }
}
