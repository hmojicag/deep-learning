package io.github.hmojicag;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;

import java.io.IOException;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws PythonExecutionException, IOException {
        System.out.println("Hello world!");


        Plot plt = Plot.create();
        plt.plot()
                .add(Arrays.asList(1.3, 2))
                .label("label")
                .linestyle("--");
        plt.xlabel("xlabel");
        plt.ylabel("ylabel");
        plt.text(0.5, 0.2, "text");
        plt.title("Title!");
        plt.legend();
        plt.show();
    }
}