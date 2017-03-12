package ml.svm;

import ml.data.DataSet;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BiFunction;

public class GradientDescentSVM2 {
    private double[] init;

    public double[] lambdas(BiFunction<double[], double[], Double> kernel,
                            DataSet d,
                            double step,
                            double C,
                            //Set<Integer> s,
                            double eps) {

        double[][] x = d.getData();
        double[] y = new double[d.getResults().length];

        for (int j = 0; j < d.getResults().length; j++) {
            if (d.getResults()[j] == 0)
                y[j] = -1;
            else
                y[j] = 1;
        }

        double[] w = init;
        System.out.println(Arrays.toString(init));
        for (int iter = 0; iter < 100000; iter++) {
            double[] curr = new double[y.length];
            double constraint = 0;
            for (int i = 0; i < curr.length; i++) {
                //if (s.contains(i)) {
                curr[i] = -1;
                for (int j = 0; j < curr.length; j++) {
                    curr[i] += y[j] * w[j] * y[i] * kernel.apply(x[i], x[j]);
                }
                w[i] -= step * curr[i];
                /*} else {
                    curr[i] = w[i];
                }*/
            }

            for (int i = 0; i < curr.length; i++) {
                double tmp = w[i] - step * curr[i];
                if (tmp < -eps)
                    curr[i] = 0;
                else if (tmp > C + eps)
                    curr[i] = C;
                else
                    curr[i] = tmp;
                constraint += curr[i] * y[i];
            }

            //System.out.println("Before " + constraint);
            for (int j = 0; j < 1000 && (Math.abs(constraint) > eps); j++) {
                double add = constraint / curr.length;
                constraint = 0;
                for (int i = 0; i < curr.length; i++) {
                    if (curr[i] - add / y[i] > -eps && curr[i] - add / y[i] < C + eps)
                        curr[i] -= add / y[i];
                    constraint += curr[i] * y[i];
                }
            }
            //System.out.println("after " + constraint);

            double tmp = 0;
            for (int i = 0; i < curr.length; i++) {
                tmp += (curr[i] - w[i]) * (curr[i] - w[i]);
            }

            if (tmp < eps * eps) {
                break;
            } else {
                w = curr;
            }
        }
        for (int i = 0; i < w.length; i++) {
            if (w[i] < eps)
                w[i] = 0;
        }
        return w;
    }


    public void initGuess(int size) {
        Random random = new Random();
        init = random.doubles(size).map(a -> a / size).toArray();
        // - 1/(2*size)
    }

}
