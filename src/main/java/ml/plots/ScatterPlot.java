package ml.plots;

import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.points.DefaultPointRenderer2D;
import de.erichseifert.gral.plots.points.PointRenderer;
import de.erichseifert.gral.ui.InteractivePanel;
import ml.data.DataSet;
import ml.primitives.Classifier;

import java.awt.*;
import java.awt.geom.Ellipse2D;


public class ScatterPlot extends ExamplePanel {
    /**
     * Version id for serialization.
     */
    private static final long serialVersionUID = -412699430625953887L;

    private static final int SAMPLE_COUNT = 100000;

    /**
     * Instance to generate random data values.
     */
    @SuppressWarnings("unchecked")
    public ScatterPlot(DataSet dataSet) {
        // Generate 100,000 data points
        DataTable goodGuys = new DataTable(Double.class, Double.class);
        DataTable badGuys = new DataTable(Double.class, Double.class);

        double[][] data1 = dataSet.getData();
        double[] results = dataSet.getResults();

        for (int i = 0; i < data1.length; i++) {
            if (results[i] >= 0.5) {
                goodGuys.add(data1[i][0], data1[i][1]);
            } else {
                badGuys.add(data1[i][0], data1[i][1]);
            }
        }

        // Create a new xy-plot
        XYPlot plot = new XYPlot(goodGuys, badGuys);
        plot.getPointRenderers(goodGuys).get(0).setColor(COLOR1);
        plot.getPointRenderers(badGuys).get(0).setColor(Color.RED);

        // Format plot
        plot.setInsets(new Insets2D.Double(-2.0, -2.0, -2.0, -2.0));
        plot.getTitle().setText(getDescription());


        // Add plot to Swing component
        add(new InteractivePanel(plot), BorderLayout.CENTER);
    }

    @SuppressWarnings("unchecked")
    public ScatterPlot(Classifier classifier, DataSet dataSet) {
        DataTable goodGuys = new DataTable(Double.class, Double.class);
        DataTable badGuys = new DataTable(Double.class, Double.class);

        for (double x = -2; x <= 2; x += 0.05) {
            for (double y = -2; y <= 2; y += 0.05) {
                if (classifier.classify(new double[]{x, y}) == 1) {
                    goodGuys.add(x, y);
                } else {
                    badGuys.add(x, y);
                }
            }
        }
        DataTable realGoodGuys = new DataTable(Double.class, Double.class);
        DataTable realBadGuys = new DataTable(Double.class, Double.class);

        double[][] data1 = dataSet.getData();
        double[] results = dataSet.getResults();

        for (int i = 0; i < data1.length; i++) {
            if (results[i] >= 0.5) {
                realGoodGuys.add(data1[i][0], data1[i][1]);
            } else {
                realBadGuys.add(data1[i][0], data1[i][1]);
            }
        }
        System.out.println("Here");
        PointRenderer points1 = new DefaultPointRenderer2D();
        points1.setShape(new Ellipse2D.Double(-3.0, -3.0, 6.0, 6.0));
        points1.setColor(new Color(0.0f, 0.3f, 1.0f, 0.3f));
        PointRenderer points2 = new DefaultPointRenderer2D();
        points2.setShape(new Ellipse2D.Double(-3.0, -3.0, 6.0, 6.0));
        points2.setColor(new Color(1.0f, 0.0f, 0.7058824f, 0.3f));

        XYPlot plot = new XYPlot(goodGuys, badGuys, realGoodGuys, realBadGuys);
        plot.setPointRenderers(goodGuys, points1);
        plot.setPointRenderers(badGuys, points2);

        plot.getPointRenderers(realGoodGuys).get(0).setColor(COLOR1);
        plot.getPointRenderers(realBadGuys).get(0).setColor(Color.RED);

        // Format plot
        plot.setInsets(new Insets2D.Double(-2.0, -2.0, -2.0, -2.0));
        plot.getTitle().setText(getDescription());


        // Add plot to Swing component
        add(new InteractivePanel(plot), BorderLayout.CENTER);

    }

    @Override
    public String getTitle() {
        return "Scatter plot";
    }

    @Override
    public String getDescription() {
        return "Kek";
    }
}