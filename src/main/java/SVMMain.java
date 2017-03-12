import ml.crossvalidation.CrossValidator;
import ml.crossvalidation.ScoreCalculator;
import ml.data.DataSet;
import ml.data.SimpleDataSet;
import ml.knn.DistanceCalc;
import ml.knn.Kernel;
import ml.knn.SimpleKNNClassifier;
import ml.primitives.Classifier;
import ml.svm.SMO;

import java.io.File;
import java.util.function.Function;

/**
 * Created by mariashka on 11/9/16.
 */
public class SVMMain {

    public static void main(String[] args) {

        DataSet dataSet = SimpleDataSet.readDataSetFromCsv(new File("chips.csv"));
//
//        double res = 0;
//        double C = 0;
//        for (double i = 33; i <= 33.1; i += 1) {
//            double finalI = i;
//            double ans = CrossValidator.tValidation(1,
//                    dataSet1 -> new SMO(dataSet1, finalI),
//                    ScoreCalculator::f1Score,
//                    dataSet,
//                    5);
//
//            if (ans > res) {
//                res = ans;
//                C = i;
//            }
//
//            System.out.println(ans);
//        }
//        System.out.print(C + "  " + res);
        Function<DataSet, Classifier> classifierSupplier = dataSet1 -> new SimpleKNNClassifier(dataSet1,
                (doubles, doubles2) -> DistanceCalc.calcMinkovskiDistance(doubles, doubles2, 1),
                Kernel::Epanechnikov, 3, 1.9);

        double v = CrossValidator.validatePValue(dataSet1 -> new SMO(dataSet1, 30),
                classifierSupplier, ScoreCalculator::f1Score, dataSet, 5, true, null);
        System.out.println(v);

        double w = CrossValidator.validateWilcoxon(dataSet1 -> new SMO(dataSet1, 30),
                classifierSupplier, ScoreCalculator::f1Score, dataSet, 5, true, null);
        System.out.println(w);
        ScoreCalculator.ConfusionMatrix confusionMatrix = CrossValidator.tValidation(
                5,
                dataSet1 -> new SMO(dataSet1, 30),
                ScoreCalculator::confusionMatrix,
                dataSet,
                5,
                ScoreCalculator.ConfusionMatrix::average);
        System.out.println(confusionMatrix);
//        Classifier classifier = new SMO(dataSet, 33);
//        ScatterPlot classifierPlot = new ScatterPlot(classifier, dataSet);
//        classifierPlot.showInFrame();
    }
}
