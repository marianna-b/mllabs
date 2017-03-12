import jzy.SurfaceDemo
import ml.data.SimpleDataSet
import ml.linear.GradientDescent
import org.jzy3d.analysis.AnalysisLauncher
import java.io.File
import java.util.*

/**
 * @author Snopi
 * 25.09.2016
 */
fun main(args: Array<String>) {
    val dataSet = SimpleDataSet.readDataSetFromCsv(File("prices.txt"))
    val data = dataSet.data
    val results = dataSet.results
    val dims = dataSet.data[0].size + 1
    val norm = DoubleArray(dims + 1)
    for (i in data.indices) {
        val datai = data[i]
        for (j in datai.indices) {
            norm[j] = Math.max(norm[j], Math.abs(datai[j]))
        }
        norm[norm.size - 1] = Math.max(norm[norm.size - 1], Math.abs(results[i]))
    }
    for (i in data.indices) {
        val datai = data[i]
        for (j in datai.indices) {
            datai[j] /= norm[j]
        }
        results[i] /= norm[norm.size - 1]
    }

    val surface = GradientDescent.getSurface({
        w: DoubleArray ->
        val res = DoubleArray(w.size)
        dataSet.data.forEachIndexed { i, doubles ->
            var tmp = w[0]
            doubles.forEachIndexed { j, d ->
                tmp += d * w[j + 1]
            }
            tmp -= dataSet.results[i]
            res[0] += tmp
            doubles.forEachIndexed { i, d ->
                res[i + 1] += d * tmp
            }
        }
        res
    }, 0.0001, dims, 1e-5)
    var ans = 0.0
    dataSet.data.forEachIndexed { i, doubles ->
        ans += (surface.apply(doubles[0], doubles[1])
                - dataSet.results[i]) * (
                surface.apply(doubles[0], doubles[1])
                        - dataSet.results[i]
                )
    }
    ans /= dataSet.results.size
    println(Math.sqrt(ans))
    val demo = SurfaceDemo(surface, dataSet)
    AnalysisLauncher.open(demo)
    while (true) {
        val scanner = Scanner(System.`in`)
        var x = scanner.nextDouble()
        var y = scanner.nextDouble()
        x /= norm[0]
        y /= norm[1]
        println(surface.apply(x, y) * norm[norm.size - 1])
    }
}
