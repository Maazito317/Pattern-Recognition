import numpy as np
import matplotlib.pylab as plt


#set the dataset
S0 = [[1, 3, 1], [2, 3, 1]]
S1 = [[-1, 1, -1], [-2, 0.5, -1]]
S1_x = []
S1_y = []
S0_x = []
S0_y = []
n = 4

for i in S0:
    S0_x.append(i[0])
    S0_y.append(i[1])
for i in S1:
    S1_x.append(i[0])
    S1_y.append(i[1])

#use step as activation function
def step(x):
    if x < 0:
        y = 0
    else:
        y = 1
    return y

def initialize_weights():
    w0_1 = np.random.uniform(-1, 1)
    w1_1 = np.random.uniform(-1, 1)
    w2_1 = np.random.uniform(-1, 1)
    W = [w0_1, w1_1, w2_1]
    return W

dataset = S0 + S1
W = initialize_weights()

def classify(test, W): #function used to classify our test vector
    z = (W[0] + (test[0] * W[1]) + (test[1] * W[2]))
    y = step(z)
    return y

def PTA(weight, learning_rate): #training algorithm does one run over dataset and updates weights accordingly
    for i in range(len(dataset)):
        z = weight[0] + (dataset[i][0] * weight[1]) + (dataset[i][1] * weight[2])
        y = step(z)
        update = [1] + dataset[i][0:2]
        desired_output = dataset[i][2]
        difference = desired_output - y
        if difference != 0:
            weight[0] = weight[0] + update[0] * learning_rate * difference
            weight[1] = weight[1] + update[1] * learning_rate * difference
            weight[2] = weight[2] + update[2] * learning_rate * difference
    return weight


learning_rate = 1
weights = PTA(W, learning_rate)
test = [1, -1]
print("Test vector classified as: ", classify(test, weights))
#classifies it as 1 as a result
