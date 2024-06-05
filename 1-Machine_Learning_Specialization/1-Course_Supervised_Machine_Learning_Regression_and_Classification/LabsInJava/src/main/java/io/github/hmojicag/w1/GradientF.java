package io.github.hmojicag.w1;

import java.util.List;
import java.util.function.Function;

public class GradientF implements Function<LinearRegressionParams, GradientResult> {
    /**
     * Computes the partial derivatives of the Cost function J(w,b) with respect to w and b
     *       dJ(w,b)        dJ(w,b)
     *         dw             db
     * @return dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
     *         dj_db (scalar): The gradient of the cost w.r.t. the parameter b
     */
    @Override
    public GradientResult apply(LinearRegressionParams params) {
        Double dj_dw = 0.0;
        Double dj_db = 0.0;
        int m = params.xValues.size();
        List<Double> xValues = params.xValues;
        List<Double> yValues = params.yValues;
        Double w = params.w;
        Double b = params.b;
        for(int i = 0; i < m; i++) {
            Double x = xValues.get(i);
            Double y = yValues.get(i);
            Double f_wb = (w * x) + b;
            dj_dw += (f_wb - y) * x;
            dj_db += f_wb - y;
        }
        dj_dw = dj_dw / m;
        dj_db = dj_db / m;
        return new GradientResult(dj_dw, dj_db);
    }
}
