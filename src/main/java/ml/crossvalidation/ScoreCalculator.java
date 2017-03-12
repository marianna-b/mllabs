package ml.crossvalidation;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * @author Snopi
 *         19.09.2016
 */
@Slf4j
public class ScoreCalculator {
    public static double simpleDoubleAverage(List<Double> doubleList) {
        return doubleList.stream().mapToDouble(k -> k).sum() / doubleList.size();
    }
    public static double accuracy(List<CrossValidator.ResultEntry> resultEntries) {
        return resultEntries.stream()
                .mapToDouble(resultEntry -> resultEntry.getExpected() == resultEntry.getGot() ? 1 : 0)
                .sum() / resultEntries.size();
    }

    public static double f1Score(List<CrossValidator.ResultEntry> resultEntries) {
        return confusionMatrix(resultEntries).getScore();
    }


    public static double mse(List<CrossValidator.ResultEntry> resultEntries) {
        double mse = 0;
        for (CrossValidator.ResultEntry resultEntry : resultEntries) {
            double diff = resultEntry.getExpected() - resultEntry.getGot();
            mse += diff * diff;
        }
        mse /= resultEntries.size();
        return mse;
    }

    public static ConfusionMatrix confusionMatrix(List<CrossValidator.ResultEntry> resultEntries) {
        int TP = 0;
        int FP = 0;
        int TN = 0;
        int FN = 0;
        for (CrossValidator.ResultEntry resultEntry : resultEntries) {
            if (resultEntry.getExpected() == 0) {
                if (resultEntry.getGot() == 0) {
                    TN++;
                } else {
                    FP++;
                }
            } else {
                if (resultEntry.getGot() == 0) {
                    FN++;
                } else {
                    TP++;
                }
            }
        }
        double score = (2.0 * TP) / (2.0 * TP + FN + FP);
        return new ConfusionMatrix(TP, FP, TN, FN, score);
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class ConfusionMatrix {
        double TP;
        double FP;
        double TN;
        double FN;
        double score;

        @Override
        public String toString() {
            return String.format("%.4f \n %4.2f %4.2f \n %4.2f %4.2f", score, TP, FN, FP, TN);
        }

        public static ConfusionMatrix average(List<ConfusionMatrix> list) {
            ConfusionMatrix res = new ConfusionMatrix();
            int l = list.size();
            for (ConfusionMatrix confusionMatrix : list) {
                res.TP += confusionMatrix.TP;
                res.FP += confusionMatrix.FP;
                res.TN += confusionMatrix.TN;
                res.FN += confusionMatrix.FN;
                res.score += confusionMatrix.score / l;
            }
            return res;
        }
    }


}
