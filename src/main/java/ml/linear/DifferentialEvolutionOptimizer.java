package ml.linear;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.function.Function;

/**
 * @author Snopi
 *         24.09.2016
 */
public class DifferentialEvolutionOptimizer {

    @Data
    @AllArgsConstructor
    public static class Agent {
        double[] weights;
        double fitness;
    }

    public static void optimize(LinearRegressor regressor, Function<LinearRegressor, Double> costFunction) {
        Random random = new Random();
        int populationSize = 75;
        double crossoverRate = 0.9;
        double differentialRate = 0.5;
        int dimensions = regressor.getWeights().length;
        int iterations = 200000;
        List<Agent> agents = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            double[] array = new double[dimensions];
            for (int j = 0; j < dimensions; j++) {
                array[j] = random.nextDouble();
            }
            regressor.setWeights(array);
            agents.add(new Agent(array, costFunction.apply(regressor)));
        }
        System.out.println(agents);
        for (int i = 0; i < iterations; i++) {
            for (int currentAgent = 0; currentAgent < populationSize; currentAgent++) {
                int optimizeIndex = random.nextInt(dimensions);
                int a = random.nextInt(populationSize);
                int b = random.nextInt(populationSize);
                while (b == a || currentAgent == b) {
                    b = random.nextInt(populationSize);
                }
                int c = random.nextInt(populationSize);
                while (a == c || c == b || c == currentAgent) {
                    c = random.nextInt(populationSize);
                }
                double[] xArray = agents.get(currentAgent).getWeights();
                double[] aArray = agents.get(a).getWeights();
                double[] bArray = agents.get(b).getWeights();
                double[] cArray = agents.get(c).getWeights();


                double[] newAgentWeights = new double[dimensions];
                for (int j = 0; j < dimensions; j++) {
                    if (optimizeIndex == j || random.nextDouble() < crossoverRate) {
                        newAgentWeights[j] = aArray[j] + differentialRate * (bArray[j] - cArray[j]);
                    } else {
                        newAgentWeights[j] = xArray[j];
                    }
                }
                regressor.setWeights(newAgentWeights);
                double newFitness = costFunction.apply(regressor);
                if (newFitness < agents.get(currentAgent).getFitness()) {
                    agents.get(currentAgent).setWeights(newAgentWeights);
                }
            }
        }

        System.out.println(agents.toString());
        Optional<Agent> agent = agents.stream().min((o1, o2) -> Double.compare(o1.getFitness(), o2.getFitness()));
        if (agent.isPresent()) {
            regressor.setWeights(agent.get().getWeights());
        }
    }
}
