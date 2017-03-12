package ml.non_param;

public class Kernels {

    public static double Rectangle(double x) {
        if (Math.abs(x) >= 1) {
            return 0;
        }
        return 0.5;
    }
    public static double Triangle(double x) {
        return Rectangle(x) * (1 - Math.abs(x));
    }
    public static double Qudratic(double x) {
        return Rectangle(x) * (1 - x * x);
    }

    public static double Quartic(double x) {
        return Rectangle(x) * (1 - x * x) * (1 - x * x);
    }

    public static double Gaussian(double x) {
        return Math.exp(-2 * x * x);
    }
}
