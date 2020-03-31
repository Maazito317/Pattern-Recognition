import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

train = 50000
test = 10000

input, output = fetch_openml('mnist_784', version=1, return_X_y=True)
print('MNIST downloaded')

train_input, test_input, train_output, test_output = train_test_split(input, output, train_size=train, test_size=test)

# Scalar used to scale the data to have zero mean and unit variance
# as needed by the Logistic regression function
scalar = StandardScaler()
train_input = scalar.fit_transform(train_input)
test_input = scalar.transform(test_input)

model = LogisticRegression(penalty='l2', C=1e3, solver='lbfgs', multi_class='multinomial', max_iter=2000)

model.fit(train_input, train_output)
sparsity = np.mean(model.coef_ == 0) * 100
score = model.score(test_input, test_output)
pred = model.predict(test_input)
print("Test score: ", score*100)
print(confusion_matrix(test_output, pred))
