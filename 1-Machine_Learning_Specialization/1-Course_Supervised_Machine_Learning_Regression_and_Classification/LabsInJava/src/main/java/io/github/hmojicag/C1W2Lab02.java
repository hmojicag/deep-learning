package io.github.hmojicag;

import io.github.hmojicag.w2.GradientDescentResultMultiVar;
import io.github.hmojicag.w2.GradientResultMultiVar;
import java.util.Arrays;
import java.util.List;

/**
 * You will use the motivating example of housing price prediction.
 * The training dataset contains three examples with four features (size, bedrooms, floors and, age)
 * shown in the table below. Note that, unlike the earlier labs, size is in sqft rather than 1000 sqft.
 * This causes an issue, which you will solve in the next lab!
 *
 * Size (sqft)	Number of Bedrooms	Number of floors	Age of Home	    Price (1000s dollars)
 *  2104	            5	               1	            45	            460
 *  1416	            3	               2	            40	            232
 *  852	                2	               1	            35	            178
 */
public class C1W2Lab02 {

    public static void main(String[] args) {
        double[] x1 = {2104, 5, 1, 45};
        double[] x2 = {1416, 3, 2, 40};
        double[] x3 = {852, 2, 1, 35};
        List<double[]> xTrainMatrix = Arrays.asList(x1, x2, x3);
        double[] yTrainVector = {460, 232, 178};
        double[] wInitial = new double[4];
        double bInitial = 0.0;
        int numIter = 1000;
        double alpha = 0.0000005;
        GradientDescentResultMultiVar result = gradientDescent(xTrainMatrix, yTrainVector, wInitial, bInitial, alpha, numIter);
        System.out.println(String.format("b,w found by gradient descent: %.4f, %s", result.b, Arrays.toString(result.w)));
        for(int i = 0; i < xTrainMatrix.size(); i++) {
            double p = predict(xTrainMatrix.get(i), result.w, result.b);
            System.out.println(String.format("prediction %.4f, target value: %.4f", p, yTrainVector[i]));
        }
    }

    /**
     * Performs batch gradient descent to learn w and b. Updates w and b by taking num_iters gradient steps with learning rate alpha
     * @param xMatrix   (m,n): Data, m examples with n features
     *                  The list has a size of m. Each element of the list is an example with n features.
     *                  Taking example in the comments of this class all elements of the list are:
     *                  0: [2104, 5, 1, 45]
     *                  1: [1416, 3, 2, 40]
     *                  2: [852, 2, 1, 35]
     * @param yValues   (m,) : target values
     * @param wInVector (n,) : initial model parameters
     * @param bIn       initial model parameters
     * @param alpha     Learning rate
     * @param numIter  number of iterations to run gradient descent
     * @return
     *          w (n,)  : Updated values of parameters
     *          b       : Updated value of parameter
     */
    private static GradientDescentResultMultiVar gradientDescent(List<double[]> xMatrix, double[] yValues, double[] wInVector,
                                                          double bIn, double alpha, int numIter) {
        double b = bIn;
        double[] w = wInVector.clone();
        for(int i = 0; i < numIter; i++) {
            // Calculate the gradient and update the parameters
            GradientResultMultiVar gradientResultMultiVar = gradient(xMatrix, yValues, w, b);
            double dj_db = gradientResultMultiVar.dj_db;
            double[] dj_dw = gradientResultMultiVar.dj_dw;

            // Update Parameters using w, b, alpha and gradient
            b = b - alpha * dj_db;
            for (int j = 0; j < w.length; j++) {
                w[j] = w[j] - alpha * dj_dw[j];
            }

            // Print cost every at intervals 10 times or as many iterations if < 10
            if(i % Math.ceil(numIter / 10) == 0) {
                double cost = cost(xMatrix, yValues, w, b);// Calculate cost
                System.out.println(String.format("Iteration %d: Cost: %.4f", i, cost));
            }
        }
        return new GradientDescentResultMultiVar(w, b);
    }

    /**
     * Computes the gradient for linear regression
     * @param xMatrix   (m,n): Data, m examples with n features
     *                  The list has a size of m. Each element of the list is an example with n features.
     *                  Taking example in the comments of this class all elements of the list are:
     *                  0: [2104, 5, 1, 45]
     *                  1: [1416, 3, 2, 40]
     *                  2: [852, 2, 1, 35]
     * @param yValues   (m,) : target values
     * @param wVector   (n,) : model parameters
     * @param b         model parameter
     * @return
     *       dj_dw (n,): The gradient of the cost w.r.t. the parameters w.
     *       dj_db:      The gradient of the cost w.r.t. the parameter b.
     */
    private static GradientResultMultiVar gradient(List<double[]> xMatrix, double[] yValues, double[] wVector, double b) {
        int m = xMatrix.size();             // Number of examples
        int n = xMatrix.get(0).length;      // Number of features
        double[] dj_dw = new double[n];
        double dj_db = 0.0;
        for (int i = 0; i < m; i++) {
            double err = dotProduct(xMatrix.get(i), wVector) + b - yValues[i];
            for (int j = 0; j < n; j++) {
                dj_dw[j] = dj_dw[j] + (err * xMatrix.get(i)[j]);
            }
            dj_db = dj_db + err;
        }
        dj_db = dj_db / m;
        dj_dw = divideAllElementsBy(dj_dw, m);
        return new GradientResultMultiVar(dj_dw, dj_db);
    }


    /**
     * Computes the cost J.
     * @param xMatrix   (m,n): Data, m examples with n features
     *                  The list has a size of m. Each element of the list is an example with n features.
     *                  Taking example in the comments of this class all elements of the list are:
     *                  0: [2104, 5, 1, 45]
     *                  1: [1416, 3, 2, 40]
     *                  2: [852, 2, 1, 35]
     * @param yValues   (m,) : target values
     * @param wVector   (n,) : model parameters
     * @param b         model parameter
     * @return The cost J
     */
    private static double cost(List<double[]> xMatrix, double[] yValues, double[] wVector, double b) {
        int m = xMatrix.size();
        double cost = 0.0;
        for(int i = 0; i < m; i++) {
            double f_wb_i = dotProduct(xMatrix.get(i), wVector) + b;
            cost += Math.pow(f_wb_i - yValues[i], 2);
        }
        cost = cost / (2 * m);
        return cost;
    }

    /**
     * Single Predict using linear regression
     * @param xVector   Example with multiple features
     * @param wVector   Model parameters
     * @param b         Model parameter
     * @return
     */
    private static double predict(double[] xVector, double[] wVector, Double b) {
        double p = 0.0;
        p += dotProduct(xVector, wVector);
        p += b;
        return p;
    }

    /**
     * Dot product of 2 vectors of same size n.
     * @param x is a vector of n elements: [x1, x2 ... xn]
     * @param w is a vector of n elements: [w1, w2 ... wn]
     * @return x1*w1 + x2*w2 + ... xn*wn
     */
    private static double dotProduct(double[] x, double[] w) {
        double p = 0.0;
        for (int i = 0; i < x.length; i++) {
            p += x[i] * w[i];
        }
        return p;
    }

    private static double[] divideAllElementsBy(double[] x, double m) {
        double[] xm = new double[x.length];
        for (int i = 0; i < xm.length; i++) {
            xm[i] = x[i] / m;
        }
        return xm;
    }

}
