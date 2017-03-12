package ml.linear;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * @author Snopi
 *         24.09.2016
 */

@Data
@AllArgsConstructor
public class LinearRegressor {
    private double[] weights;
    private int size;

    public double assume(double[] points) {
        double ans = weights[size];
        for (int i = 0; i < points.length; i++) {
            ans += points[i] * weights[i];
        }
//        ans += weights[size];
        return ans;
    }
}
