package ml.non_param;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.StandardXYItemRenderer;
import org.jfree.chart.renderer.xy.VectorRenderer;
import org.jfree.chart.renderer.xy.XYDotRenderer;
import org.jfree.chart.renderer.xy.XYSplineRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;

import javax.swing.*;
import java.awt.*;

public class Plot2D extends ApplicationFrame {

    private XYPlot plot;

    private int datasetIndex = 0;

    public Plot2D(final String title) {
        super(title);
        final XYSeriesCollection dataset1 = new XYSeriesCollection();
        final JFreeChart chart = ChartFactory.createScatterPlot(
            "", "X", "Y", dataset1, PlotOrientation.VERTICAL, true, true, false
        );
        chart.setBackgroundPaint(Color.white);

        this.plot = chart.getXYPlot();
        this.plot.setBackgroundPaint(Color.lightGray);
        this.plot.setDomainGridlinePaint(Color.white);
        this.plot.setRangeGridlinePaint(Color.white);
        final ValueAxis axis = this.plot.getDomainAxis();
        axis.setAutoRange(true);

        final JPanel content = new JPanel(new BorderLayout());

        final ChartPanel chartPanel = new ChartPanel(chart);
        content.add(chartPanel);

        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
        setContentPane(content);
    }

    public void addDataset(String name, double[] x, double[] y) {
        final XYSeries series = new XYSeries(name);
        for (int i = 0; i < x.length; i++) {
            series.add(x[i], y[i]);
        }

        this.datasetIndex++;
        this.plot.setDataset(
            this.datasetIndex, new XYSeriesCollection(series)
        );
        this.plot.setRenderer(this.datasetIndex, new StandardXYItemRenderer());
    }
}
