package ml.svd;

import org.jetbrains.annotations.Contract;

import java.util.Arrays;
import java.util.OptionalDouble;
import java.util.Random;

/**
 * @author Snopi
 *         03.12.2016
 */
public class SVD {

    private double λ1;
    private double λ2;
    private double μ;
    private double γ1;
    private double γ2;
    private int f;
    private double[] b_u;
    private double[] b_i;
    private double[][] q;
    private double[][] p;

    public static double[][] generateRandomMatrix(int w, int h, double bound) {
        Random random = new Random();
        double[][] tmp = new double[h][];
        for (int i = 0; i < h; i++) {
            tmp[i] = random.doubles().limit(w).map(operand -> operand * 2 * bound - bound).toArray();
        }
        return tmp;
    }

    public SVD(double λ1, double λ2, double μ, double γ1, double γ2, int userCount, int itemCount, int f) {
        this.λ1 = λ1;
        this.λ2 = λ2;
        this.μ = μ;
        this.γ1 = γ1;
        this.γ2 = γ2;
        this.b_u = new double[userCount];
        this.b_i = new double[itemCount];
        this.f = f;
        double bound = 1.0 / f;
        this.p = generateRandomMatrix(f, userCount, bound);
        this.q = generateRandomMatrix(f, itemCount, bound);
    }

    public double predict(int userId, int itemId) {
        return μ + b_i[itemId] + b_u[userId] + dotProduct(q[itemId], p[userId]);
    }

    public double gradientStep(int userId, int itemId, double rating) {
        double e_ui = rating - predict(userId, itemId);
        double diff_b_u = γ1 * (e_ui - λ1 * b_u[userId]);
        double diff_b_i = γ1 * (e_ui - λ1 * b_i[itemId]);
        double[] diff_q_i = generateDiffForStep(γ2, λ2, e_ui, p[userId], q[itemId]);
        double[] diff_p_u = generateDiffForStep(γ2, λ2, e_ui, q[itemId], p[userId]);
        OptionalDouble aDouble = Arrays.stream(new double[]{diff_b_i, diff_b_u, norm(diff_p_u), norm(diff_q_i)}).
                map(Math::abs).max();
        double maxDiff = aDouble.orElseGet(() -> 0);
        for (int i = 0; i < f; i++) {
            p[userId][i] += diff_p_u[i];
            q[itemId][i] += diff_q_i[i];
        }
        b_i[itemId] += diff_b_i;
        b_u[userId] += diff_b_u;
        return maxDiff;
    }

    @org.jetbrains.annotations.Contract(pure = true)
    private double[] generateDiffForStep(double γ2, double λ2, double e_ui, double[] d1, double[] d2) {
        double[] tmpArray = new double[f];
        for (int i = 0; i < tmpArray.length; i++) {
            tmpArray[i] = γ2 * (e_ui * d1[i] - λ2 * d2[i]);
        }
        return tmpArray;
    }

    @Contract(pure = true)
    public static double dotProduct(double[] x, double[] y) {
        double tmp = 0;
        for (int i = 0; i < x.length; i++) {
            tmp += x[i] * y[i];
        }
        return tmp;
    }

    public static double norm(double[] x) {
        return Math.sqrt(Arrays.stream(x).map(operand -> operand * operand).sum());
    }
}
