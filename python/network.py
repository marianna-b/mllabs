import random

import numpy as np


# http://neuralnetworksanddeeplearning.com/chap1.html
# https://github.com/mnielsen/neural-networks-and-deep-learning
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivation(z):
    return sigmoid(z) * (1 - sigmoid(z))


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


class Network(object):
    def __init__(self, layers):
        self.layers = layers
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def recognize(self, a):
        return np.argmax(self.recognition_results(a))

    def recognition_results(self, a):
        a = np.reshape(a, (784, 1))
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def predict(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def predict_digit(self, a):
        return np.argmax(self.predict(a))

    def validate(self, test_data):
        test_results = [(np.argmax(self.predict(x)), y) for (x, y) in test_data]
        wrongs = []
        for x, y in test_data:
            res = self.predict_digit(x)
            if res != y:
                wrongs.append((x, y, res))
        return sum(int(x == y) for (x, y) in test_results), wrongs

    def train(self, x, y, c):
        tmp_b, tmp_w = self.backpropagation(x, y)
        self.weights = [w - c * nw for w, nw in zip(self.weights, tmp_w)]
        self.biases = [b - c * nb for b, nb in zip(self.biases, tmp_b)]

    def fit(self, training_data, iterations, batch_size, eta):
        for j in range(iterations):
            random.shuffle(training_data)

            for batch in [training_data[k: k + batch_size] for k in range(0, len(training_data), batch_size)]:
                for x, y in batch:
                    tmp_b, tmp_w = self.backpropagation(x, y)
                    self.weights = [w - (eta / batch_size) * nw for w, nw in zip(self.weights, tmp_w)]
                    self.biases = [b - (eta / batch_size) * nb for b, nb in zip(self.biases, tmp_b)]

    def backpropagation(self, x, y):
        diff_b = [np.zeros(b.shape) for b in self.biases]
        diff_w = [np.zeros(w.shape) for w in self.weights]

        activations = [x]
        weighted_ins = []

        for b, w in zip(self.biases, self.weights):
            w_in = np.dot(w, activations[-1]) + b
            weighted_ins.append(w_in)
            activations.append(sigmoid(w_in))

        err = 0
        for l in range(1, len(self.layers)):
            if l == 1:
                err = (activations[-1] - vectorized_result(y)) * sigmoid_derivation(weighted_ins[-1])
            else:
                err = np.dot(self.weights[1 - l].transpose(), err) * sigmoid_derivation(weighted_ins[-l])
            diff_b[-l] = err
            diff_w[-l] = np.dot(err, activations[-l - 1].transpose())

        return diff_b, diff_w

    def serialize_to_file(self, filename: str):
        import gzip
        import pickle
        with gzip.open(filename, 'wb') as f:
            pickle.dump(self, f)


def deserialize_from_file(filename: str) -> Network:
    import gzip
    import pickle
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)
