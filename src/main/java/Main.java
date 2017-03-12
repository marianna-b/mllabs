import jzy.SurfaceDemo;
import lombok.extern.slf4j.Slf4j;
import ml.data.DataSet;
import ml.data.SimpleDataSet;
import ml.linear.DifferentialEvolutionOptimizer;
import ml.linear.LinearRegressor;
import org.jzy3d.analysis.AnalysisLauncher;

import java.io.File;
import java.util.Scanner;
import java.util.function.Function;

/**
 * Created by snopi on 9/18/16.
 */


@Slf4j
public class Main {
    public static void main(String[] args) throws Exception {

        DataSet dataSet = SimpleDataSet.readDataSetFromCsv(new File("prices.txt"));
        int dims = dataSet.getData()[0].length;
        LinearRegressor linearRegressor = new LinearRegressor(new double[dims + 1], dims);

        //normalization
        double[] norm = new double[dims + 1];
        double[][] data = dataSet.getData();
        double[] results = dataSet.getResults();

        for (int i = 0; i < data.length; i++) {
            double[] datai = data[i];
            for (int j = 0; j < datai.length; j++) {
                norm[j] = Math.max(norm[j], Math.abs(datai[j]));
            }
            norm[norm.length - 1] = Math.max(norm[norm.length - 1], Math.abs(results[i]));
        }

        for (int i = 0; i < data.length; i++) {
            double[] datai = data[i];
            for (int j = 0; j < datai.length; j++) {
                datai[j] /= norm[j];
            }
            results[i] /= norm[norm.length - 1];
        }


        Function<LinearRegressor, Double> costFunction = regressor -> {
            double ans = 0;
            for (int i = 0; i < data.length; i++) {
                ans += Math.pow(regressor.assume(data[i]) - results[i], 2);
            }
            ans /= data.length;
            ans = Math.sqrt(ans);
            return ans;
        };
        DifferentialEvolutionOptimizer.optimize(linearRegressor, costFunction);
        System.out.println(costFunction.apply(linearRegressor));

        SurfaceDemo demo = new SurfaceDemo((a, b) -> linearRegressor.assume(new double[]{a, b}), dataSet);
        AnalysisLauncher.open(demo);
        //noinspection InfiniteLoopStatement
        while (true) {
            Scanner scanner = new Scanner(System.in);
            double x = scanner.nextDouble();
            double y = scanner.nextDouble();
            x /= norm[0];
            y /= norm[1];
            System.out.println(linearRegressor.assume(new double[]{x, y}) * norm[norm.length - 1]);
        }
    }
}
