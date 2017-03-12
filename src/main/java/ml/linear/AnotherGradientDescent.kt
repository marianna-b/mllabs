package ml.linear

import java.util.*

/**
 * @author Snopi
 * 25.09.2016
 */

fun getSurface(problemSize: Long,
               lossFuncDerivation: (weights : List<Double>) -> List<Double>,
               step: Double,
               eps: Double,
               weightFun: (points : List<Double>, weights : List<Double>) -> Double
): (Double, Double) -> Double {

    val random = Random()
    var current: List<Double> = random.doubles(problemSize.toLong()).toArray().asList() //weights
    val maxIterations: Int = 100000

    for (iteration in 0..maxIterations) {
        val derivation = lossFuncDerivation(current)
        if (derivation.sum() < eps) {
            break
        }
        current = current.zip(derivation) { x, y ->
            x - step * y
        }
    }

    return { a: Double, b: Double ->
        weightFun(listOf(a, b), current)
    }
}
