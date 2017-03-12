package jmath;

import org.math.plot.Plot2DPanel;
import org.math.plot.plotObjects.BaseLabel;

import javax.swing.*;
import java.awt.*;

import static java.lang.Math.PI;
import static org.math.array.StatisticSample.randomNormal;
import static org.math.array.StatisticSample.randomUniform;

public class CustomPlotExample {
    public static void main(String[] args) {

        // define your data
        double[] x = randomNormal(1000, 0, 1); // 1000 random numbers from a normal (Gaussian) statistical law
        double[] y = randomUniform(1000, -3, 3); // 1000 random numbers from a uniform statistical law

        // create your PlotPanel (you can use it as a JPanel)
        Plot2DPanel plot = new Plot2DPanel();
        plot.addScatterPlot("kek", x, y);

        JFrame frame = new JFrame("a plot panel");
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);

    }
}