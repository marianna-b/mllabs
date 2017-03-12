package ml.svm;

import ml.knn.DistanceCalc;

import java.util.function.BiFunction;

/**
 * @author Snopi
 *         13.11.2016
 */
public class SVMKernels {
    public static double RBF(double[] x, double[] y) {
        double dist = DistanceCalc.calcEuclidDistance(x, y);
        return Math.exp(-dist * dist) / 2;
    };
}
