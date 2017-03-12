package ml.non_param

/**
 * @author Snopi
 * 17.11.2016
 */

val eps = 1e-6
val maxIters = 15

data class Point(val x: Double, val y: Double)

fun getGammas(trainSet: List<Point>,
              kernel1: (Double) -> Double,
              kernel2: (Double) -> Double,
              k: Int): List<Double> {

    var currentGammas: List<Double> = DoubleArray(trainSet.size) { 1.0 }.asList()
    var iter = 0
    do {
        val windowSizes: List<Double> = trainSet.mapIndexed { i, point ->
            trainSet.filterIndexed { j, _i -> i != j }
                    .map { p -> Math.abs(p.x - point.x) }
                    .sorted()[k]
        }

        val a: List<Double> = trainSet.mapIndexed { i, dataI ->
            val pairs = trainSet.zip(currentGammas)
                    .filterIndexed { j, pair -> i != j }
                    .map {
                        val (p, γ) = it
                        val (x, y) = p
                        Pair(γ * kernel1((x - dataI.x) / windowSizes[i]), y)
                    }
            pairs.map { it.first * it.second }.sum() / pairs.map { it.first }.sum()
        }

        val ε = a.zip(trainSet.map { it.y }).map { Math.abs(it.first - it.second) }
        val robust = ε.sorted()[ε.size / 2] * 6
        val nextGammas = ε.map { kernel2(it / robust) }
        val diff = nextGammas.zip(currentGammas) { x, y -> x - y }.map { it * it }.sum()
        currentGammas = nextGammas
        iter++
        if (iter > maxIters)
            break
    } while (diff > eps)
    return currentGammas
}