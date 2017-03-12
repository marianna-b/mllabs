import ml.crossvalidation.CrossValidator;
import ml.crossvalidation.ScoreCalculator;
import ml.data.DataSet;
import ml.data.SimpleDataSet;
import ml.knn.Kernel;
import ml.non_param.*;
import ml.primitives.Classifier;
import org.jfree.ui.RefineryUtilities;
import org.math.plot.Plot2DPanel;

import javax.swing.*;
import java.io.File;
import java.util.*;
import java.util.function.Function;


public class NonParametric {

    public static List<Point> kek(DataSet dataSet) {
        List<Point> pointList = new ArrayList<>();
        for (int i = 0; i < dataSet.getResults().length; i++) {
            pointList.add(new Point(dataSet.getData()[i][1], dataSet.getResults()[i]));
        }
        return pointList;
    }

    public static void main(String[] args) throws Exception {
        DataSet dataSet = SimpleDataSet.readDataSetFromCsv(new File("non-parametric.csv"));


        Map<String, Function<Double, Double>> kernels = new HashMap<>();
//        kernels.put("Rectangle", Kernels::Rectangle);
//        kernels.put("Triangle", Kernels::Triangle);
//        kernels.put("Qudratic", Kernels::Qudratic);
//        kernels.put("Quartic", Kernels::Quartic);
//        kernels.put("Gaussian", Kernels::Gaussian);

        kernels = Kernel.kernelMap;
        Function<DataSet, Classifier> classifierSupplier;
        Function<DataSet, Classifier> bestSupplWithW = null;
        Function<DataSet, Classifier> bestSuppl = null;

        for (Map.Entry<String, Function<Double, Double>> entry : kernels.entrySet()) {
            double H = 0;
            double ans = 1000000;
            for (double h = 1; h < 3; h += 0.1) {
                double finalH = h;
                classifierSupplier = dataSet1 -> new KernelSmoothing(dataSet1, finalH, entry.getValue());
                double v = CrossValidator.validate(classifierSupplier, ScoreCalculator::mse, dataSet, 5, true,
                        ScoreCalculator::simpleDoubleAverage);
                if (v < ans) {
                    ans = v;
                    bestSuppl = classifierSupplier;
                    H = h;
                }
            }
            System.out.println(entry.getKey() + " " + ans + " with " + H);
        }

        double[] x = new double[dataSet.getResults().length];
        for (int i = 0; i < dataSet.getData().length; i++) {
            x[i] = dataSet.getData()[i][1];
        }


        double WIDTH = 60;
        int UPPER_BOUND = 2000;
        double step = WIDTH / UPPER_BOUND;
        double[] x1 = new double[UPPER_BOUND];
        double[] y1 = new double[UPPER_BOUND];
        double[] y2 = new double[UPPER_BOUND];

//        Classifier kernelSmoothing = new KernelSmoothing(dataSet, 4.3, Kernel::Quartic);
        Classifier kernelSmoothing = bestSuppl.apply(dataSet);
        Classifier lowess = new KernelSmoothing(dataSet, 2.8, Kernel::Triweight,
                LowessRegressionKt.getGammas(kek(dataSet),
                        Kernel::Triweight,
                        Kernel::Quartic,
                        12).stream()
                        .mapToDouble(a -> a)
                        .toArray());
//        kernelSmoothing = lowess;

        double currentX = 0;
        for (int i = 0; i < UPPER_BOUND; i++) {
            x1[i] = currentX;
            double[] tmp = new double[2];
            tmp[1] = currentX;
            y1[i] = kernelSmoothing.classify(tmp);
            y2[i] = lowess.classify(tmp);
            currentX += step;
        }

        Plot2D demo = new Plot2D("Nadaraya Watson Regression");
        demo.addDataset("LOWESS", x1, y2);
        demo.addDataset("Classified", x1, y1);
        demo.addDataset("Dataset Points", x, dataSet.getResults());
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);


        Plot2DPanel plot = new Plot2DPanel();
        plot.addLinePlot("KEEEEK", x1, y2);
        plot.addLinePlot("kek", x1, y1);
        plot.addScatterPlot("kok", x, dataSet.getResults());
        JFrame frame = new JFrame("a plot panel");
        frame.setSize(800, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);
    }

}
