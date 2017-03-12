package ml.linear;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Created by mariashka on 9/25/16.
 */
public class GradientDescent {

    public static BiFunction<Double, Double, Double> getSurface(Function<double[], double[]> gradQ,
                                                  double step,
                                                  int size,
                                                  double eps) {

        double[] w = initGuess(size);
        for (int iter = 0; iter < 100000; iter++) {
            double[] curr = gradQ.apply(w);

            for (int i = 0; i < curr.length; i++) {
                curr[i] = step * curr[i];
            }

            double tmp = 0;
            for (double aCurr : curr) {
                tmp += aCurr * aCurr;
            }

            if (tmp < eps * eps) {
                break;
            } else {
                for (int k = 0; k < curr.length; k++) {
                    w[k] -= curr[k];
                }
            }
        }
        return (x, y) ->  w[0] + x * w[1] + y * w[2];
    }

    private static double[] initGuess(int size) {
        Random random = new Random();
        return random.doubles(size).map(a -> a/(400.0) - 1/(200.0)).toArray();
    }


}
