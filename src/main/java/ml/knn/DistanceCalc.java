package ml.knn;

/**
 * @author Snopi
 *         18.09.2016
 */
public class DistanceCalc {

    public static double calcMinkovskiDistance(double[] x, double[] y, double p) {
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            sum += Math.pow(Math.abs(y[i] - x[i]), p);
        }
        return Math.pow(sum, 1 / p);
    }

    public static double calcEuclidDistance(double[] x, double[] y) {
        return calcMinkovskiDistance(x, y, 2);
    }

    public static double calcManhattanDistance(double[] x, double[] y) {
        return calcMinkovskiDistance(x, y, 1);
    }
}
