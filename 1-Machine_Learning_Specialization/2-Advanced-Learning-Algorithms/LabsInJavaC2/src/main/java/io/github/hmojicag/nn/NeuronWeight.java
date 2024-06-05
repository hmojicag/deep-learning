package io.github.hmojicag.nn;

public class NeuronWeight {
    private double[] w;
    private double b;

    public NeuronWeight(double[] w, double b) {
        this.w = w;
        this.b = b;
    }

    public double[] getW() {
        return w;
    }

    public void setW(double[] w) {
        this.w = w;
    }

    public double getB() {
        return b;
    }

    public void setB(double b) {
        this.b = b;
    }
}
