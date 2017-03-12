package ml.data;

import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.dataformat.csv.CsvMapper;
import com.fasterxml.jackson.dataformat.csv.CsvParser;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.SneakyThrows;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Snopi
 *         18.09.2016
 */

@Data
@AllArgsConstructor
public class SimpleDataSet implements DataSet {
    private double[][] data;
    private double[] results;

    @SneakyThrows
    public static DataSet readDataSetFromCsv(File csvFile) {
        CsvMapper mapper = new CsvMapper();
        mapper.enable(CsvParser.Feature.WRAP_AS_ARRAY);
        MappingIterator<double[]> it = mapper.readerFor(double[].class).readValues(csvFile);
        double[][] all = it.readAll().toArray(new double[][]{});
        double[] results = new double[all.length];

        List<double[]> newAll = new ArrayList<>();
        for (int i = 0; i < all.length; i++) {
            results[i] = all[i][all[i].length - 1];
            newAll.add(Arrays.copyOfRange(all[i], 0, all[i].length - 1));
        }

        return new SimpleDataSet(newAll.toArray(new double[][]{}), results);
    }
}
