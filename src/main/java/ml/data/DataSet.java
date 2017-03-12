package ml.data;

/**
 * @author Snopi
 *         18.09.2016
 */
public interface DataSet {
    double[][] getData();
    double[] getResults();
    default double[] getRow(int index) {
        return getData()[index];
    }
}
