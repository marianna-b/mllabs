import lombok.SneakyThrows;
import ml.crossvalidation.CrossValidator;
import ml.crossvalidation.ScoreCalculator;
import ml.data.DataSet;
import ml.data.SimpleDataSet;
import ml.knn.DistanceCalc;
import ml.knn.Kernel;
import ml.knn.SimpleKNNClassifier;
import ml.plots.ScatterPlot;
import ml.primitives.Classifier;

import java.io.File;
import java.util.Map;
import java.util.function.Function;

/**
 * @author Snopi
 *         24.09.2016
 */
public class KNNMain {
    @SneakyThrows
    public static void main(String[] args) {
        DataSet dataSet = SimpleDataSet.readDataSetFromCsv(new File("chips.csv"));
        System.out.println(dataSet);

//        ml.plots.ScatterPlot scatterPlot = new ml.plots.ScatterPlot(dataSet);
//        scatterPlot.showInFrame();

        Function<DataSet, Classifier> classifierSupplier = dataSet1 -> new SimpleKNNClassifier(dataSet1,
                (doubles, doubles2) -> DistanceCalc.calcMinkovskiDistance(doubles, doubles2, 1),
                Kernel::Epanechnikov, 4, 2.9);

        double ans = CrossValidator.tValidation(5,
                classifierSupplier,
                ScoreCalculator::f1Score,
                dataSet,
                5);

        System.out.println(ans);

        Classifier classifier = classifierSupplier.apply(dataSet);

        ScatterPlot classifierPlot = new ScatterPlot(classifier, dataSet);
        classifierPlot.showInFrame();
//        bruteForce(dataSet);
    }

    public static void bruteForce(DataSet dataSet) {
        double maxScore = 0;
        for (int neighbors = 1; neighbors < 30; neighbors++) {
            for (Map.Entry<String, Function<Double, Double>> entry : Kernel.kernelMap.entrySet()) {
                for (int metric = 1; metric <= 3; metric++) {
                    for (double width = 0.5; width <= 3; width += 0.1) {
                        int finalNeighbors = neighbors;
                        double finalWidth = width;
                        final double finalMetric = metric;
                        int iters = 5;
                        double validate = CrossValidator.tValidation(iters,
                                dataSet1 -> new SimpleKNNClassifier(dataSet1,
                                        (doubles, doubles2) ->
                                                DistanceCalc.calcMinkovskiDistance(doubles, doubles2, finalMetric),
                                        entry.getValue(), finalNeighbors, finalWidth),
                                ScoreCalculator::f1Score,
                                dataSet,
                                5);

                    if (validate > maxScore) {
                        maxScore = validate;
                        System.out.format("score: %s | %s, %s, %s, %s\n", maxScore,
                                neighbors,
                                entry.getKey(),
                                width,
                                metric);
                    }
                }
            }
        }
    }
}
}
