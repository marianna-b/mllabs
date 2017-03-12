package ml.crossvalidation;

import java.util.List;

/**
 * @author Snopi
 *         14.11.2016
 */
public interface Score<T> {
    T getScore();
    T fold(List<T> scores);
}
