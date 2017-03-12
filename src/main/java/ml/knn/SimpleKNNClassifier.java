package ml.knn;

import lombok.AllArgsConstructor;
import ml.data.DataSet;
import ml.primitives.Classifier;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * @author mariashka
 *         18.09.2016
 */
@AllArgsConstructor
public class SimpleKNNClassifier implements Classifier {
    private DataSet dataSet;
    private BiFunction<double[], double[], Double> metric;
    private Function<Double, Double> kernel;
    private int k;
    private double h;

    @Override
    public double classify(double[] x) {
        Comparator<Integer> c = (t1, t2) -> {
            Double res1 = metric.apply(x, dataSet.getRow(t1));
            Double res2 = metric.apply(x, dataSet.getRow(t2));
            if (res1 < res2)
                return -1;
            else if (res1 > res2)
                return 1;
            else
                return 0;
        };
        PriorityQueue<Integer> q = new PriorityQueue<>(c);
        for (int i = 0; i < dataSet.getData().length; i++) {
            q.add(i);
        }
        double res0 = 0;
        double res1 = 0;
        double[] results = dataSet.getResults();
        for (int i = 0; i < k; i++) {
            int curr = q.poll();
            Double res = metric.apply(x, dataSet.getRow(curr));
            if (results[curr] == 0)
                res0 += kernel.apply(res / h);
            else
                res1 += kernel.apply(res / h);
        }
        if (res0 > res1)
            return 0;
        else
            return 1;
    }
}
