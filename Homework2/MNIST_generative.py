import numpy as np
from matplotlib import pyplot as plt
import idx2numpy
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

train_input = np.asfarray(idx2numpy.convert_from_file('train-images-idx3-ubyte'))
train_output = np.asfarray(idx2numpy.convert_from_file('train-labels-idx1-ubyte'))
test_input = np.asfarray(idx2numpy.convert_from_file('t10k-images-idx3-ubyte'))
test_output = np.asfarray(idx2numpy.convert_from_file('t10k-labels-idx1-ubyte'))

pixel_num = 28 * 28
label_num = 10
train_input = train_input.reshape(60000, 784)
train_output = train_output.reshape(60000, 1)
test_input = test_input.reshape(10000, 784)
test_output = test_output.reshape(10000, 1)

print(train_input.dtype, train_output.dtype)
print(train_input.shape, train_output.shape)

X_train, X_val, y_train, y_val = train_test_split(train_input, train_output, test_size=1 / 6,
                                                  random_state=4)  # Create Training and validation sets: X is the img, y is the label
cls = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
cls.fit(X_train, y_train)  ##Fit the bernoulli model

print(cls.classes_)
pi = cls.class_count_ / 50000  # prior probabilities
print(pi)

pred = cls.predict(X_val)  # classify validation set
print("Validation accuracy = ", metrics.accuracy_score(y_val, pred))

test_pred = cls.predict(test_input)
print("Test error = ", 1 - metrics.accuracy_score(test_output, test_pred))

p = np.random.permutation(len(test_input))
p = p[:30]

index = 0
misclassified = []
for label, predict in zip(test_pred, test_output):
    if label != predict:
        misclassified.append(index)
    index += 1

plt.figure(figsize=(20, 4))
for i in range(0, 5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(np.reshape(test_input[misclassified[i]], (28, 28)), cmap=plt.cm.gray)
    plt.title('Actual: {}, Predicted: {}'.format(test_pred[misclassified[i]], test_output[misclassified[i]]),
              fontsize=20)

plt.show()