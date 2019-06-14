import random
import numpy as np

# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix

"""
def plot_accuracy(train_accuracy_per_epoch, validation_accuracy_per_epoch, epoch_count):
    plt.plot(range(1, epoch_count + 1), train_accuracy_per_epoch, label="train")
    plt.plot(range(1, epoch_count + 1), validation_accuracy_per_epoch, label="validation")
    plt.xlabel("epoch number")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

"""


class DataSet:
    def __init__(self, file_path, is_test_set=False, batch_size=1, image_shape=None):
        if image_shape is None:
            image_shape = [3, 32, 32]
        self._file_path = file_path
        self._is_test_set = is_test_set
        self._batch_size = batch_size
        self._lines_offsets = None
        self._lines_order = None
        self._current_line = None
        self._image_shape = image_shape
        self._construct_lines()

    def shuffle(self):
        random.shuffle(self._lines_order)

    def __len__(self):
        return len(self._lines_order)

    def __iter__(self):
        self._current_line = 0
        return self

    def __next__(self):
        if self._current_line >= len(self):
            raise StopIteration()

        with open(self._file_path) as data_set_file:
            data = []

            for _ in range(self._batch_size):
                if self._current_line >= len(self):
                    break

                data_set_file.seek(self._lines_offsets[self._lines_order[self._current_line]])
                data_example = data_set_file.readline().strip("?, \n\t\r").split(',')
                data.append(data_example)
                self._current_line += 1

            data = np.array(data, dtype=np.float32)
            if self._is_test_set:
                return data.reshape([self._batch_size] + self._image_shape)

            classification = data[:, 0].astype("uint8")
            return np.delete(data, [0], axis=1).reshape([self._batch_size] + self._image_shape), classification

    def _construct_lines(self):
        self._lines_offsets = []
        self._lines_order = []
        self._current_line = 0

        with open(self._file_path, "rb") as f:
            offset = 0

            for index, line in enumerate(f):
                self._lines_order.append(index)
                self._lines_offsets.append(offset)
                offset += len(line)


def save_test_prediction(output_path, test_predictions):
    with open(output_path, "w") as output_file:
        output_file.write("\n".join(map(str, test_predictions)))


class Layer:
    def __init__(self):
        self.is_training = True
        self.learning_rate = 0.1

    def forward(self, inputs):
        raise NotImplementedError()

    def backward(self, inputs, grad_output):
        raise NotImplementedError()


class ActivationLayer(Layer):
    def forward(self, inputs):
        raise NotImplementedError()

    def backward(self, inputs, grad_output):
        raise NotImplementedError()


class ReLU(ActivationLayer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return np.maximum(0, inputs)

    def backward(self, inputs, grad_output):
        return grad_output * (inputs > 0)


class LReLU(ActivationLayer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self._alpha = alpha

    def forward(self, inputs):
        return np.maximum(0, inputs)

    def backward(self, inputs, grad_output):
        der_inputs = np.ones_like(inputs)
        der_inputs[inputs < 0] = self._alpha
        return grad_output * der_inputs


class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return np.where(inputs >= 0,
                        1 / (1 + np.exp(-inputs)),
                        np.exp(inputs) / (1 + np.exp(inputs)))

    def backward(self, inputs, grad_output):
        return grad_output * self.forward(inputs) * (1 - self.forward(inputs))


class Dense(Layer):
    def __init__(self, input_units, output_units):
        super().__init__()
        self._weights = np.random.normal(scale=np.sqrt(2 / (input_units + output_units)),
                                         size=(input_units, output_units))
        self._biases = np.zeros(output_units)

    def forward(self, inputs):
        return np.dot(inputs, self._weights) + self._biases

    def backward(self, inputs, grad_output):
        grad_input = np.dot(grad_output, self._weights.T)

        grad_weights = np.dot(inputs.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * inputs.shape[0]

        self._weights = self._weights - self.learning_rate * grad_weights
        self._biases = self._biases - self.learning_rate * grad_biases

        return grad_input


class Dropout(Layer):
    def __init__(self, dropout_rate):
        super().__init__()
        self._dropout_rate = dropout_rate
        self._dropout = None

    def forward(self, inputs):
        if not self.is_training:
            return inputs

        self._dropout = np.random.rand(*inputs.shape) > self._dropout_rate
        return self._dropout * inputs * 1 / (1 - self._dropout_rate)

    def backward(self, inputs, grad_output):
        if not self.is_training:
            return grad_output

        return self._dropout * grad_output * 1 / (1 - self._dropout_rate)


class BatchNorm(Layer):
    def __init__(self, input_size):
        super().__init__()
        self._gamma = np.ones((1, input_size))
        self._beta = np.zeros((1, input_size))
        self._var = None
        self._mu = None
        self._test_var = None
        self._test_mu = None

    def forward(self, inputs):
        self._mu = np.mean(inputs, axis=0)
        self._var = np.var(inputs, axis=0)

        if self.is_training:
            inputs_norm = (inputs - self._mu) / np.sqrt(self._var + 1e-8)
        else:
            inputs_norm = (inputs - self._test_mu) / np.sqrt(self._test_var + 1e-8)

        return self._gamma * inputs_norm + self._beta

    def backward(self, inputs, grad_output):
        inputs_mu = inputs - self._mu
        std_inv = 1. / np.sqrt(self._var + 1e-8)
        inputs_norm = inputs_mu * std_inv

        dgamma = np.sum(grad_output * inputs_norm, axis=0)
        dbeta = np.sum(grad_output, axis=0)
        dinputs_norm = grad_output * self._gamma
        dvar = np.sum(dinputs_norm * inputs_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.sum(dinputs_norm * -std_inv, axis=0) + dvar * np.mean(-2. * inputs_mu, axis=0)

        self._gamma -= self.learning_rate * dgamma
        self._beta -= self.learning_rate * dbeta

        if self._test_var is None:
            self._test_var = self._var

        if self._test_mu is None:
            self._test_mu = self._mu

        self._test_var = 0.9 * self._test_var + 0.1 * self._var
        self._test_mu = 0.9 * self._test_mu + 0.1 * self._mu

        return (dinputs_norm * std_inv) + (dvar * 2 * inputs_mu / inputs.shape[0]) + (dmu / inputs.shape[0])


class Conv2D(Layer):
    def __init__(self, channels_in, channels_out, filter_size):
        super().__init__()
        self._channels_in = channels_in
        self._channels_out = channels_out
        self._filter_size = filter_size
        self._filters = np.random.normal(scale=np.sqrt(2 / ((self._filter_size ** 2) * self._channels_in)),
                                         size=(self._channels_out, self._channels_in, filter_size, filter_size))

    def forward(self, inputs):
        batch_size, channels_in, x, y = inputs.shape
        assert self._channels_in == channels_in, \
            "incorrect channel count passed to forward (%d != %d)" % (channels_in, self._channels_in)

        assert x == y, "Only squared inputs are supported (x=%d != y=%d)" % (x, y)
        assert x >= self._filter_size, \
            "Inputs size is too small for the filter size (x=%d < f=%d)" % (x, self._filter_size)

        x_out = y_out = x - self._filter_size + 1

        output = np.zeros([batch_size, self._channels_out, x_out, y_out])
        for h in range(0, x_out):
            for v in range(0, y_out):
                inputs_slice = inputs[:, None, :, h: h + self._filter_size, v:v + self._filter_size]
                output[:, :, h, v] = np.sum(inputs_slice * self._filters, axis=(2, 3, 4))

        return output

    def backward(self, inputs, grad_output):
        batch_size, channels_out, x_out, y_out = grad_output.shape

        gradient_inputs = np.zeros_like(inputs)
        gradient_filters = np.zeros_like(self._filters)

        for h in range(0, x_out):
            for v in range(0, y_out):
                gradient_inputs[:, :, h:h + self._filter_size, v:v + self._filter_size] += \
                    np.sum(self._filters * grad_output[:, :, None, h, v], axis=1)
                gradient_filters += np.sum(
                    (inputs[:, None, :, h:h + self._filter_size, v:v + self._filter_size] *
                     grad_output[:, :, None, h, v]), axis=0)
        return gradient_inputs


def grad_softmax_cross_entropy(inputs, correct_ids):
    ones_for_correct = np.zeros_like(inputs)
    ones_for_correct[np.arange(len(inputs)), correct_ids] = 1
    exponents = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
    softmax = exponents / exponents.sum(axis=-1, keepdims=True)

    return (-ones_for_correct + softmax) / inputs.shape[0]


class NeuralNetwork(object):

    def __init__(self, class_id_map, layers, training_set, validation_set, epoch_count, learning_rate=0.1):
        self._epoch_count = epoch_count
        self._class_id_map = class_id_map
        self._id_class_map = {v: k for k, v in self._class_id_map.items()}
        self._layers = layers
        self._training_set = training_set
        self._validation_set = validation_set
        self._learning_rate = learning_rate
        self._learning_rate_step = self._learning_rate * 0.5 / self._epoch_count

    def predict(self, inputs):
        self._test_mode()
        net_classifications = np.vectorize(lambda x: self._id_class_map[x])(self._forward(inputs)[-1].argmax(axis=-1))
        self._training_mode()
        return net_classifications

    def train(self):
        train_accuracy_per_epoch = list()
        validation_accuracy_per_epoch = list()

        for _ in range(self._epoch_count):
            self._training_set.shuffle()
            train_correct_times_count = 0
            for train_examples, classifications in self._training_set:
                layer_activations = [train_examples] + self._forward(train_examples)
                correct_ids = np.vectorize(lambda x: self._class_id_map[x])(classifications)

                train_correct_times_count += \
                    np.sum(correct_ids == layer_activations[-1].argmax(axis=-1))

                self._backward(layer_activations, correct_ids)

            self._learning_rate -= self._learning_rate_step

            validation_correct_times_count = 0
            for data, classifications in self._validation_set:
                net_classifications = self.predict(data)
                validation_correct_times_count += np.sum(classifications == net_classifications)

            train_accuracy = float(train_correct_times_count) / len(self._training_set)
            validation_accuracy = float(validation_correct_times_count) / len(self._validation_set)
            train_accuracy_per_epoch.append(train_accuracy)
            validation_accuracy_per_epoch.append(validation_accuracy)
            print(train_accuracy, validation_accuracy)
        """
        plot_accuracy(train_accuracy_per_epoch, validation_accuracy_per_epoch, self._epoch_count)
        predictions = list()
        real_classes = list()
        for data, classifications in self._validation_set:
            predictions.extend(self.predict(data))
            real_classes.extend(classifications)
        print(confusion_matrix(real_classes, predictions))
        """

    def _forward(self, inputs):
        activations = []
        for layer in self._layers:
            activations.append(layer.forward(inputs))
            inputs = activations[-1]

        return activations

    def _backward(self, layer_activations, correct_ids):
        grad = grad_softmax_cross_entropy(layer_activations[-1], correct_ids)
        for layer_index in reversed(range(len(self._layers))):
            grad = self._layers[layer_index].backward(layer_activations[layer_index], grad)

    def _training_mode(self):
        for layer in self._layers:
            layer.is_training = True

    def _test_mode(self):
        for layer in self._layers:
            layer.is_training = False

    @property
    def _learning_rate(self):
        return self._layers[0].learning_rate

    @_learning_rate.setter
    def _learning_rate(self, learning_rate):
        for layer in self._layers:
            layer.learning_rate = learning_rate


def main():
    test_set = DataSet("data/test.csv", is_test_set=True)
    train_set = DataSet("data/train.csv", batch_size=64)
    validation_set = DataSet("data/validate.csv", batch_size=64)

    layers = list()
    layers.append(Conv2D(3, 10, 4))
    layers.append(Dense(3072, 512))
    layers.append(BatchNorm(512))
    layers.append(LReLU())
    layers.append(Dropout(0.5))
    layers.append(Dense(512, 64))
    layers.append(BatchNorm(64))
    layers.append(LReLU())
    layers.append(Dropout(0.5))
    layers.append(Dense(64, 10))

    nn = NeuralNetwork({x: x - 1 for x in range(1, 11)}, layers, train_set, validation_set, 40)
    nn.train()

    test_predictions = []
    for data in test_set:
        test_predictions.extend(nn.predict(data))

    save_test_prediction("output.txt", test_predictions)


if __name__ == "__main__":
    main()
