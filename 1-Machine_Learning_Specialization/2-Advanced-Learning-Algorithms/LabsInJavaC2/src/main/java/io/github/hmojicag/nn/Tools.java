package io.github.hmojicag.nn;

public class Tools {
    /**
     * Dot product of 2 vectors of same size n.
     * @param x is a vector of n elements: [x1, x2 ... xn]
     * @param w is a vector of n elements: [w1, w2 ... wn]
     * @return x1*w1 + x2*w2 + ... xn*wn
     */
    public static double dotProduct(double[] x, double[] w) {
        double p = 0.0;
        for (int i = 0; i < x.length; i++) {
            p += x[i] * w[i];
        }
        return p;
    }
}
