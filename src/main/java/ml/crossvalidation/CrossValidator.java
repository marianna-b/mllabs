package ml.crossvalidation;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import ml.data.DataSet;
import ml.data.SimpleDataSet;
import ml.primitives.Classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @author Snopi
 *         18.09.2016
 */
public class CrossValidator {

    @AllArgsConstructor
    @Data
    public static class PairWithResult {
        public double[] x;
        public double y;
    }

    @AllArgsConstructor
    @Data
    public static class Fold {
        List<PairWithResult> fold;
    }

    @AllArgsConstructor
    @NoArgsConstructor
    @Data
    public static class ResultEntry {
        double expected;
        double got;
    }

    public static <T> T tValidation(int t,
                                     Function<DataSet, Classifier> classifierSupplier,
                                     Function<List<ResultEntry>, T> scoreCalculator,
                                     DataSet dataSet,
                                     int k,
                                     Function<List<T>, T> resultAverage) {
        List<T> answers = new ArrayList<>();
        for (int i = 0; i < t; i++) {

            T res = CrossValidator.validate(
                    classifierSupplier,
                    scoreCalculator,
                    dataSet,
                    5, true, resultAverage);
//            System.out.println(tmp);
            answers.add(res);
        }
        return resultAverage.apply(answers);
    }

    public static double tValidation(int t,
                                    Function<DataSet, Classifier> classifierSupplier,
                                    Function<List<ResultEntry>, Double> scoreCalculator,
                                    DataSet dataSet,
                                    int k) {
        return tValidation(t, classifierSupplier, scoreCalculator, dataSet, k, ScoreCalculator::simpleDoubleAverage);
    }

    public static <T> T validate(Function<DataSet, Classifier> classifierSupplier,
                                      Function<List<ResultEntry>, T> scoreCalculator,
                                      DataSet dataSet,
                                      int k,
                                      boolean shuffle,
                                      Function<List<T>, T> resultAverage) {
        double[][] data = dataSet.getData();
        double[] results = dataSet.getResults();

        List<PairWithResult> allData = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            allData.add(new PairWithResult(data[i], results[i]));
        }
        if (shuffle) {
            Collections.shuffle(allData);
        }

        List<Fold> folds = new ArrayList<>();
        int foldSize = data.length / k;
        for (int i = 0; i < k; i++) {
            folds.add(new Fold(allData.subList(i * foldSize, (i + 1) * foldSize)));
        }
        double score = 0;
        List<T> resList = new ArrayList<>();

        for (Fold fold : folds) {
            resList.add(checkOnCurrentFold(folds, fold, classifierSupplier, scoreCalculator));
        }
        return resultAverage.apply(resList);
    }

    private static <T> T checkOnCurrentFold(List<Fold> folds,
                                            Fold fold,
                                            Function<DataSet, Classifier> classifierSupplier,
                                            Function<List<ResultEntry>, T> scoreCalculator) {
        List<PairWithResult> collect = folds.stream()
                .filter(fold1 -> fold1 != fold) //never do this
                .flatMap(f -> f.getFold().stream())
                .collect(Collectors.toList());

        double[] results = new double[collect.size()];
        double[][] data = new double[collect.size()][];
        for (int i = 0; i < collect.size(); i++) {
            data[i] = collect.get(i).getX();
            results[i] = collect.get(i).getY();
        }

        DataSet dataSet = new SimpleDataSet(data, results);
        Classifier classifier = classifierSupplier.apply(dataSet);

        List<ResultEntry> resultEntries = fold.getFold().stream()
                .map(pairWithResult -> new ResultEntry(pairWithResult.getY(),
                        classifier.classify(pairWithResult.getX())))
                .collect(Collectors.toList());

        return scoreCalculator.apply(resultEntries);
    }

    public static double validatePValue(Function<DataSet, Classifier> classifierSupplier1,
                                      Function<DataSet, Classifier> classifierSupplier2,
                                      Function<List<ResultEntry>, Double> scoreCalculator,
                                      DataSet dataSet,
                                      int k,
                                      boolean shuffle,
                                      Function<List<Double>, Double> resultAverage) {
        double[][] data = dataSet.getData();
        double[] results = dataSet.getResults();

        List<PairWithResult> allData = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            allData.add(new PairWithResult(data[i], results[i]));
        }
        if (shuffle) {
            Collections.shuffle(allData);
        }

        List<Fold> folds = new ArrayList<>();
        int foldSize = data.length / k;
        for (int i = 0; i < k; i++) {
            folds.add(new Fold(allData.subList(i * foldSize, (i + 1) * foldSize)));
        }
        double score = 0;
        List<Double> resList1 = new ArrayList<>();
        List<Double> resList2 = new ArrayList<>();

        for (Fold fold : folds) {
            resList1.add(checkOnCurrentFold(folds, fold, classifierSupplier1, scoreCalculator));
            resList2.add(checkOnCurrentFold(folds, fold, classifierSupplier2, scoreCalculator));
        }
        double pValue = 0;
        for (int i = 0; i < resList1.size(); i++) {
            double x = resList1.get(i);
            double y = resList2.get(i);
            pValue += (x - y) * (x - y) / x;
        }
        return pValue;
    }

    public static double validateWilcoxon(Function<DataSet, Classifier> classifierSupplier1,
                                      Function<DataSet, Classifier> classifierSupplier2,
                                      Function<List<ResultEntry>, Double> scoreCalculator,
                                      DataSet dataSet,
                                      int k,
                                      boolean shuffle,
                                      Function<List<Double>, Double> resultAverage) {
        double[][] data = dataSet.getData();
        double[] results = dataSet.getResults();

        List<PairWithResult> allData = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            allData.add(new PairWithResult(data[i], results[i]));
        }
        if (shuffle) {
            Collections.shuffle(allData);
        }

        List<Fold> folds = new ArrayList<>();
        int foldSize = data.length / k;
        for (int i = 0; i < k; i++) {
            folds.add(new Fold(allData.subList(i * foldSize, (i + 1) * foldSize)));
        }
        double score = 0;
        List<Double> resList1 = new ArrayList<>();
        List<Double> resList2 = new ArrayList<>();

        for (Fold fold : folds) {
            resList1.add(checkOnCurrentFold(folds, fold, classifierSupplier1, scoreCalculator));
            resList2.add(checkOnCurrentFold(folds, fold, classifierSupplier2, scoreCalculator));
        }
        double pValue = 0;
        List<Pair> listOfPairs = new ArrayList<>();
        for (int i = 0; i < resList1.size(); i++) {
            double x = resList1.get(i);
            double y = resList2.get(i);

            double dif = Math.abs(x - y);
            if (dif != 0) {
                listOfPairs.add(new Pair(x, y, dif));
            }
        }
        Collections.sort(listOfPairs, (pair, t1) -> t1.absDif < pair.absDif ? 1 : -1);
        double res = 0;
//        System.out.println(listOfPairs.toString());
        for (int i = 0; i < listOfPairs.size(); i++) {
            double sign = Math.signum(listOfPairs.get(i).x - listOfPairs.get(i).y);
            res += (i + 1) * sign;
        }
        return res;
    }
}
@Data
@AllArgsConstructor
@NoArgsConstructor
class Pair {
    double x;
    double y;
    double absDif;
}