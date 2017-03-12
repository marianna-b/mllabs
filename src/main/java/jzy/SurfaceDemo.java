package jzy;

import ml.data.DataSet;
import org.jzy3d.analysis.AbstractAnalysis;
import org.jzy3d.chart.factories.AWTChartComponentFactory;
import org.jzy3d.colors.Color;
import org.jzy3d.colors.ColorMapper;
import org.jzy3d.colors.colormaps.ColorMapRainbow;
import org.jzy3d.maths.Coord3d;
import org.jzy3d.maths.Range;
import org.jzy3d.plot3d.builder.Builder;
import org.jzy3d.plot3d.builder.Mapper;
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid;
import org.jzy3d.plot3d.primitives.Scatter;
import org.jzy3d.plot3d.primitives.Shape;
import org.jzy3d.plot3d.rendering.canvas.Quality;

import java.util.function.BiFunction;

public class SurfaceDemo extends AbstractAnalysis {

    private BiFunction<Double, Double, Double> linearRegressor;

    public SurfaceDemo(BiFunction<Double, Double, Double> linearRegressor, DataSet dataSet) {
        this.linearRegressor = linearRegressor;
        this.dataSet = dataSet;
    }

    private DataSet dataSet;
    @Override
    public void init() {
        // Define a function to plot
        Mapper mapper = new Mapper() {
            @Override
            public double f(double x, double y) {
                return linearRegressor.apply(x, y);
            }
        };

        // Define range and precision for the function to plot
        Range range = new Range(0, 1);
        int steps = 80;

        // Create the object to represent the function over the given range.
        final Shape surface = Builder.buildOrthonormal(new OrthonormalGrid(range, steps, range, steps), mapper);
        surface.setColorMapper(new ColorMapper(new ColorMapRainbow(), surface.getBounds().getZmin(), surface.getBounds().getZmax(), new Color(1, 1, 1, 200f)));
        surface.setFaceDisplayed(true);
        surface.setWireframeDisplayed(false);

        // Create a chart
        chart = AWTChartComponentFactory.chart(Quality.Nicest, getCanvasType());
        chart.getScene().getGraph().add(surface);

        Coord3d[] points = new Coord3d[dataSet.getData().length];
        Color[] colors = new Color[dataSet.getData().length];
        for (int i = 0; i < points.length; i++) {
            double[] datai = dataSet.getData()[i];
            points[i] = new Coord3d(datai[0], datai[1], dataSet.getResults()[i]);
            colors[i] = new Color(0, 0, 0);
        }

        Scatter scatter = new Scatter(points, colors);
        scatter.setWidth(10);
        chart.getScene().getGraph().add(scatter);
    }
}