package ml.non_param;

import ml.data.DataSet;
import ml.primitives.Classifier;

import java.util.Arrays;
import java.util.function.Function;
import java.util.stream.DoubleStream;

public class KernelSmoothing implements Classifier {

    private DataSet dataSet;
    private double h;
    private Function<Double, Double> kernel;
    private double[] weights;

    public KernelSmoothing(DataSet dataSet, double h,
                           Function<Double, Double> kernel) {

        this(dataSet, h, kernel, DoubleStream.generate(() -> 1).limit(dataSet.getResults().length).toArray());
    }

    public KernelSmoothing(DataSet dataSet, double h,
                           Function<Double, Double> kernel, double[] weights) {
        this.dataSet = dataSet;
        this.h = h;
        this.kernel = kernel;
        this.weights = weights;
    }

    public double classify(double[] x) {
        double num = 0;
        double denom = 0;
        for (int i = 0; i < dataSet.getResults().length; i++) {
            double kern = weights[i] * kernel.apply((x[1] - dataSet.getData()[i][1]) / h);
            num += dataSet.getResults()[i] * kern;
            denom += kern;
        }
        if (denom == 0) {
            return 0;
        }
        return num / denom;
    }
}
