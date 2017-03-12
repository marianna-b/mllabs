package ml.svm;

import lombok.extern.slf4j.Slf4j;
import ml.data.DataSet;
import ml.primitives.Classifier;

import java.util.concurrent.ThreadLocalRandom;
import java.util.function.BiFunction;

@Slf4j
public class SMO implements Classifier {

    private double[][] trainPoints;
    private double[] trainAnswers;
    private double[] lambdas;
    private double scalarShift = 0;

    @SuppressWarnings("FieldCanBeLocal")
    private int dimensionOfProblem;
    private int sizeOfTrainSet;

    private BiFunction<double[], double[], Double> kernel;

    private double eps = 1e-8;
    @SuppressWarnings("FieldCanBeLocal")
    private double tolerance = 2 * 1e-4;
    private double upperBoundOfLambdas;

    public SMO(DataSet dataSet, double upperBoundOfLambdas, BiFunction<double[], double[], Double> kernel) {
        this.upperBoundOfLambdas = upperBoundOfLambdas;

        trainPoints = dataSet.getData();
        trainAnswers = new double[dataSet.getResults().length];

        sizeOfTrainSet = trainAnswers.length;
        dimensionOfProblem = dataSet.getRow(0).length;

        for (int j = 0; j < sizeOfTrainSet; j++) {
            if (dataSet.getResults()[j] == 0) {
                trainAnswers[j] = -1;
            } else {
                trainAnswers[j] = 1;
            }
        }

        this.kernel = kernel;
        fit();
    }

    public SMO(DataSet dataSet, double upperBoundOfLambdas) {
        this(dataSet, upperBoundOfLambdas, SVMKernels::RBF);
    }


    private int examineExample(int j) {
        double[] jTrainPoint = trainPoints[j];
        double jTrainAnswer = trainAnswers[j];
        double jError = calcError(jTrainPoint, jTrainAnswer);
        double jLambda = lambdas[j];

        if ((jTrainAnswer * jError < -tolerance && jLambda < upperBoundOfLambdas)
                || (jTrainAnswer * jError > tolerance && jLambda > 0)) {
            int idx = 0;
            double res = -1000000000;
            int amount = 0;
            for (int i = 0; i < sizeOfTrainSet; i++) {
                if (lambdas[i] > 0 && lambdas[i] < upperBoundOfLambdas)
                    amount++;
            }

            if (amount > 1) {
                for (int i = 0; i < sizeOfTrainSet; i++) {
                    double tmp = Math.abs(calcError(trainPoints[i], trainAnswers[i]) - jError);
                    if (res < tmp) {
                        res = tmp;
                        idx = i;
                    }
                }
                if (takeStep(idx, j) == 1)
                    return 1;
            }

            int m = ThreadLocalRandom.current().nextInt(0, sizeOfTrainSet);
            for (int i = 0; i < sizeOfTrainSet; i++) {
                idx = (i + m) % sizeOfTrainSet;
                if (lambdas[idx] > 0 && lambdas[idx] < upperBoundOfLambdas)
                    if (takeStep(idx, j) == 1)
                        return 1;
            }

            m = ThreadLocalRandom.current().nextInt(0, sizeOfTrainSet);
            for (int i = 0; i < sizeOfTrainSet; i++) {
                idx = (i + m) % sizeOfTrainSet;
                if (takeStep(idx, j) == 1)
                    return 1;
            }
            boolean b = false;
            for (int i = 0; i < sizeOfTrainSet; i++) {
                if (takeStep(i, j) == 1) {
                    b = true;
                }
            }
            if (b) {
                return 1;
            }
        }
        return 0;
    }

    private double clip(double leftBound, double rightBound, double value) {
        if (value < leftBound) return leftBound;
        if (value > rightBound) return rightBound;
        return value;
    }

    private double cropToBordersWithMargin(double leftBound, double rightBound, double ε, double value) {
        if (value < leftBound + ε)
            return 0;
        if (value > rightBound - ε)
            return rightBound;
        return value;
    }

    private int takeStep(int i, int j) {
        if (i == j) {
            return 0;
        }
        double[] iPoint = trainPoints[i];
        double iAnswer = trainAnswers[i];
        double iError = calcError(iPoint, iAnswer);
        double iLambda = lambdas[i];

        double[] jPoint = trainPoints[j];
        double jAnswer = trainAnswers[j];
        double jError = calcError(jPoint, jAnswer);
        double jLambda = lambdas[j];

        double lowBound = calcLowBound(jLambda, iLambda, jAnswer, iAnswer);
        double highBound = calcHighBound(jLambda, iLambda, jAnswer, iAnswer);

        if (Math.abs(lowBound - highBound) < eps) {
            return 0;
        }

        double η = 2 * kernel.apply(iPoint, jPoint)
                - kernel.apply(iPoint, iPoint)
                - kernel.apply(jPoint, jPoint);

        if (η >= 0) {
            log.error("------------------------------------------------------------");
            return 0;
        }

        lambdas[j] = jLambda - (jAnswer * (iError - jError)) / η;
        lambdas[j] = clip(lowBound, highBound, lambdas[j]);
        lambdas[j] = cropToBordersWithMargin(0, upperBoundOfLambdas, 1e-8, lambdas[j]);

        if (Math.abs(lambdas[j] - jLambda) < eps * (lambdas[j] + jLambda + eps)) {
            lambdas[j] = jLambda; //undo changes
            return 0;
        }

        lambdas[i] = iLambda + iAnswer * jAnswer * (jLambda - lambdas[j]);

        double b1 = iError + iAnswer * (lambdas[i] - iLambda) * kernel.apply(iPoint, iPoint)
                + jAnswer * (lambdas[j] - jLambda) * kernel.apply(iPoint, jPoint)
                + scalarShift;

        double b2 = jError
                + iAnswer * (lambdas[i] - iLambda) * kernel.apply(iPoint, jPoint)
                + jAnswer * (lambdas[j] - jLambda) * kernel.apply(jPoint, jPoint)
                + scalarShift;

        if (lambdas[i] < 0 || lambdas[i] > upperBoundOfLambdas) {
            scalarShift = b1;
        } else if (lambdas[j] < 0 || lambdas[j] > upperBoundOfLambdas) {
            scalarShift = b2;
        } else {
            scalarShift = (b1 + b2) / 2;
        }

        return 1;
    }


    private void fit() {
        lambdas = new double[sizeOfTrainSet];
        scalarShift = 0;
        int numChanged = 0;
        boolean examineAll = true;
        while (numChanged > 0 || examineAll) {
            numChanged = 0;

            for (int i = 0; i < sizeOfTrainSet; i++) {
                if (examineAll || lambdas[i] > 0 && lambdas[i] < upperBoundOfLambdas) {
                    numChanged += examineExample(i);
                }
            }

            if (examineAll) {
                examineAll = false;
            } else if (numChanged == 0) {
                examineAll = true;
            }
        }

//        System.out.println(Arrays.toString(lambdas));
//        System.out.println(scalarShift);
    }

    private double calcHighBound(double jLambda, double iLambda, double jAnswer, double iAnswer) {
        if (Math.abs(iAnswer - jAnswer) > eps)
            return Math.min(upperBoundOfLambdas, upperBoundOfLambdas - iLambda + jLambda);
        else
            return Math.min(upperBoundOfLambdas, iLambda + jLambda);
    }

    private double calcLowBound(double jLambda, double iLambda, double jAnswer, double iAnswer) {
        if (Math.abs(iAnswer - jAnswer) > eps) {
            return Math.max(0, jLambda - iLambda);
        } else {
            return Math.max(0, iLambda + jLambda - upperBoundOfLambdas);
        }
    }

    private double calcError(double[] xj, double yj) {
        return predict(xj) - yj;
    }

    private int predict(double[] xj) {
        double curr = -scalarShift;
        for (int i = 0; i < sizeOfTrainSet; i++) {
            curr += lambdas[i] * trainAnswers[i] * kernel.apply(xj, trainPoints[i]);
        }
        if (curr >= 0)
            return 1;
        else
            return -1;
    }

    public double classify(double[] x) {
        if (predict(x) >= 0)
            return 1;
        else
            return 0;
    }

}
