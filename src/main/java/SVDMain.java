import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.SneakyThrows;
import ml.svd.SVD;
import org.simpleflatmapper.csv.CsvParser;
import org.simpleflatmapper.csv.CsvWriter;
import org.simpleflatmapper.util.CheckedConsumer;
import org.simpleflatmapper.util.CloseableIterator;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * @author Snopi
 *         03.12.2016
 */
public class SVDMain {
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class TrainSetEntry {
        public int user;
        public int item;
        public double rating;
    }

    @AllArgsConstructor
    @Data
    @NoArgsConstructor
    public static class PredictionResult {
        public int Id;
        public double Prediction;
    }

    @SneakyThrows
    public static void main(String[] args) {
        long l = System.currentTimeMillis();
//        List<TrainSetEntry> trainSet = new ArrayList<>(100_000_000);
        double λ = 0.0219;
        double γ = 0.00533;
        SVD svd = new SVD(λ, λ, 3.6, γ, γ, 2649430, 17772, 25);

        File file = new File("C:\\train.csv");
        System.out.println(file.exists());

        int cnt = 0;
        for (int j = 0; j < 10; j++) {
            try (CloseableIterator<String[]> it = CsvParser.skip(1).iterator(file)) {
                while (it.hasNext()) {
                    String[] next = it.next();
//                trainSet.add(new TrainSetEntry(
//                        Integer.parseInt(next[0]),
//                        Integer.parseInt(next[1]),
//                        (double) Integer.parseInt(next[2])));
                    int userId = Integer.parseInt(next[0]);
                    int itemId = Integer.parseInt(next[1]);
                    double rating = Integer.parseInt(next[2]);
                    svd.gradientStep(userId,
                            itemId,
                            rating
                    );
                    cnt++;
                    if (cnt % 1000000 == 0) {
                        System.out.println("Already Read " + cnt);
                    }
                }
            }
        }

        List<PredictionResult> predictionResults = new ArrayList<>();


        try (CloseableIterator<String[]> it = CsvParser.skip(1).iterator(new File("C:\\test-ids.csv"))) {
            while (it.hasNext()) {
                String[] next = it.next();
                double predict = svd.predict(Integer.parseInt(next[1]),
                        Integer.parseInt(next[2]));
                predictionResults.add(new PredictionResult(Integer.parseInt(next[0]), predict));
                cnt++;
                if (cnt % 1000000 == 0) {
                    System.out.println("Already Read " + cnt);
                }
            }
        }

        writeCsv(predictionResults, new File("some-submission.csv"));

//        System.out.println(System.currentTimeMillis() - l);
//        System.out.println(trainSet.size());
//        double μ = trainSet.stream().mapToDouble(TrainSetEntry::getRating).sum() / trainSet.size();

//        cnt = 0;
//        for (TrainSetEntry trainSetEntry : trainSet) {
//            svd.gradientStep(trainSetEntry.getUser(), trainSetEntry.getItem(), trainSetEntry.rating);
//            cnt++;
//            System.out.println(cnt);
//        }

    }

    public static void writeCsv(Collection<PredictionResult> objects, File file)
            throws IOException {
        CsvWriter.CsvWriterDSL<PredictionResult> writerDsl =
                CsvWriter.from(PredictionResult.class).columns("Id", "Prediction");

        try (FileWriter fileWriter = new FileWriter(file)) {
            CsvWriter<PredictionResult> writer =
                    writerDsl.to(fileWriter);
            objects.forEach(CheckedConsumer.toConsumer(writer::append));
        }
    }
}
