# Credit: https://github.com/christianversloot/machine-learning-articles/blob/main/linking-maths-and-intuition-rosenblatts-perceptron-in-python.md?plain=1

import numpy as np
import pandas as pd

iris = pd.read_csv("Iris.csv")
print(iris['Species'].value_counts())

## Prepare a dataset

# Generate target classes {0, 1}
zeros = np.zeros(50)
ones = zeros + 1
targets = np.concatenate((zeros, ones))

# print(zeros)
# print(ones)
# print(targets)

# Generate data
# normal has attributes mu, sigma, output shape
iris_classes = iris[iris['Species'].isin(['Iris-setosa','Iris-versicolor','Iris-virginica'])] # only these 2 species
#X = iris_classes[['SepalLengthCm']].values
#Y = iris_classes[['PetalLengthCm']].values
X = iris_classes[['SepalLengthCm','PetalLengthCm']].values   # two features


B = (iris_classes['Species'] == 'Iris-setosa') .astype(int).values

### Visualizing the dataset

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
#plt.title('Perceptron on Iris (Setosa vs Versicolor)')
#plt.xlabel('Sepal Length (cm)')
#plt.ylabel('Petal Length (cm)')
#plt.scatter(X[T==0], Y[T==0], color='blue',label='Setosa (T=0)')
#plt.scatter(X[T==1], Y[T==1], color='red',label='Versicolor(T=1)')
#plt.show()



# Rosenblatt Perceptron

import numpy as np

# Basic Rosenblatt Perceptron implementation
class RBPerceptron():

  # Constructor object, self is the instance of the object NeuralNetwork itself
  def __init__(self, number_of_epochs = 100, learning_rate = 0.1):
    self.number_of_epochs = number_of_epochs
    self.learning_rate = learning_rate


  # Train perceptron
  def train(self, X, T):
    # Initialize weights vector with zeroes
    num_features = X.shape[1]
    self.w = np.zeros(num_features + 1)
    # Perform the epochs
    diff = []
    err = []
    for i in range(self.number_of_epochs):
      err.append(sum(diff))
      # For every combination of (X_i, T_i), zip creates the tuples
      for sample, desired_outcome in zip(X, T):
        # Generate prediction and compare with desired outcome
        prediction    = self.predict(sample)
        difference    = (desired_outcome - prediction)
#        diff.append(difference)
        # Compute weight update via Perceptron Learning Rule
        self.w[1:]    += self.learning_rate * difference * sample
        self.w[0]     += self.learning_rate * difference * 1
    return self

  # Generate prediction
  def predict(self, sample):
    # dot product:
    outcome = np.dot(sample, self.w[1:]) + self.w[0]
    # Activation function:
    return np.where(outcome > 0, 1, 0)

# ks = [100,200,300,400,500,600,700, 800, 900, 1000]
ks = [100,200, 300, 500]
colors = ['blue','limegreen','gray','cyan','red','red','red']

for k in ks:
  # print(k)
  rbp = RBPerceptron(k, 0.1)
  trained_model = rbp.train(X, B)

  plot_decision_regions(X, B.astype(int), clf=trained_model, legend=0)

plt.title('Perceptron')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()


# now for (Sepal Width, Petal Width) as your two features and as before (Setosa, Versicolor) as your two classes
T = (iris_classes['Species'] == 'Iris-virginica') .astype(int).values
Z = iris_classes[['SepalWidthCm','PetalWidthCm']].values   # two features
print(len(Z))

trained_model = rbp.train(Z, T)

plot_decision_regions(Z, T.astype(int), clf=trained_model, legend=0)

plt.title('Perceptron')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()