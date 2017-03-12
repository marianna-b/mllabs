package ml.knn;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * @author Snopi
 *         18.09.2016
 */
public class Kernel {

    public static Map<String, Function<Double, Double>> kernelMap = new HashMap<>();

    static {
        kernelMap.put("Epanechnikov", Kernel::Epanechnikov);
        kernelMap.put("Uniform", Kernel::Uniform);
        kernelMap.put("Gaussian", Kernel::Gaussian);
        kernelMap.put("Triweight", Kernel::Triweight);
        kernelMap.put("Quartic", Kernel::Quartic);
    }

    public static double Uniform(double x) {
        if (Math.abs(x) >= 1) {
            return 0;
        }
        return 0.5;
    }

    public static double Epanechnikov(double x) {
        if (Math.abs(x) >= 1) {
            return 0;
        }
        return 3.0 / 4 * (1 - x * x);
    }

    public static double Quartic(double x) {
        if (Math.abs(x) >= 1) {
            return 0;
        }
        return 15.0 / 16 * Math.pow(1 - x * x, 2);
    }

    public static double Triweight(double x) {
        if (Math.abs(x) >= 1) {
            return 0;
        }
        return 35.0 / 32 * Math.pow(1 - x * x, 3);
    }

    public static double Gaussian(double x) {
        if (Math.abs(x) >= 1) {
            return 0;
        }
        return 1.0 / (Math.sqrt(Math.PI*2)) * Math.pow(Math.E, -0.5*x*x);
    }
}
