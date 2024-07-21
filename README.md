## 1. Import packages


```python
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from utils import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)
```


```python
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
```

## 2. Load and process data


```python
## Visualize an image in our dataset
index = 19
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
```

    y = 1. It's a cat picture.
    


    
![png](DNN_Cat_image_classification_files/DNN_Cat_image_classification_4_1.png)
    



```python
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))
```

    Number of training examples: 209
    Number of testing examples: 50
    Each image is of size: (64, 64, 3)
    train_x_orig shape: (209, 64, 64, 3)
    train_y shape: (1, 209)
    test_x_orig shape: (50, 64, 64, 3)
    test_y shape: (1, 50)
    


```python
# Reshape the training and test examples 

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
```

    train_x's shape: (12288, 209)
    test_x's shape: (12288, 50)
    

## 3. Model architecture

* L-layer Neural Network
* 2-layers Neural Netwrork
#### Steps: 
1. Initialize parameters / Define hyperparameters
2. Loop for num_iterations:
    a. Forward propagation
    b. Compute cost function
    c. Backward propagation
    d. Update parameters (using parameters, and grads from backprop) 
3. Use trained parameters to predict labels

Helper functions from utils.py:
```python
def initialize_parameters(n_x, n_h, n_y):
    ...
    return parameters 
def linear_activation_forward(A_prev, W, b, activation):
    ...
    return A, cache
def compute_cost(AL, Y):
    ...
    return cost
def linear_activation_backward(dA, cache, activation):
    ...
    return dA_prev, dW, db
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
```



```python
### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h = 7         
n_y = 1          
layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0075
```


```python
## 2-Layer model
def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Gradient descent
    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        
        # Compute cost
        cost = compute_cost(A2, Y)
        
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
```


```python
parameters, costs = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=False)
plot_costs(costs, learning_rate)
```

    Cost after iteration 2499: 0.04421498215868956
    


    
![png](DNN_Cat_image_classification_files/DNN_Cat_image_classification_10_1.png)
    



```python
predictions_train = predict(train_x, train_y, parameters)
```

    Accuracy: 0.9999999999999998
    


```python
predictions_test = predict(test_x, test_y, parameters)
```

    Accuracy: 0.72
    

### L_layer_model 

Helper functions from utils.py:
```python
def initialize_parameters_deep(layers_dims):
    ...
    return parameters 
def L_model_forward(X, parameters):
    ...
    return AL, caches
def compute_cost(AL, Y):
    ...
    return cost
def L_model_backward(AL, Y, caches):
    ...
    return grads
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
```


```python
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """ 
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    
    return parameters, costs
```


```python
parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1, print_cost = False)

print("Cost after first iteration: " + str(costs[0]))
```

    Cost after iteration 0: 0.6950464961800915
    Cost after first iteration: 0.6950464961800915
    


```python
parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
plot_costs(costs, learning_rate)
```

    Cost after iteration 0: 0.6950464961800915
    Cost after iteration 100: 0.5892596054583805
    Cost after iteration 200: 0.5232609173622991
    Cost after iteration 300: 0.4497686396221906
    Cost after iteration 400: 0.4209002161883899
    Cost after iteration 500: 0.37246403061745953
    Cost after iteration 600: 0.34742051870201895
    Cost after iteration 700: 0.31719191987370277
    Cost after iteration 800: 0.2664377434774657
    Cost after iteration 900: 0.21991432807842554
    Cost after iteration 1000: 0.14357898893623783
    Cost after iteration 1100: 0.45309212623221284
    Cost after iteration 1200: 0.09499357670093515
    Cost after iteration 1300: 0.08014128076781371
    Cost after iteration 1400: 0.06940234005536462
    Cost after iteration 1500: 0.06021664023174592
    Cost after iteration 1600: 0.05327415758001877
    Cost after iteration 1700: 0.04762903262098433
    Cost after iteration 1800: 0.04297588879436869
    Cost after iteration 1900: 0.03903607436513818
    Cost after iteration 2000: 0.03568313638049027
    Cost after iteration 2100: 0.03291526373054675
    Cost after iteration 2200: 0.030472193059120623
    Cost after iteration 2300: 0.02838785921294613
    Cost after iteration 2400: 0.026615212372776073
    Cost after iteration 2499: 0.02482129221835338
    


    
![png](DNN_Cat_image_classification_files/DNN_Cat_image_classification_16_1.png)
    



```python
pred_train = predict(train_x, train_y, parameters)
```

    Accuracy: 0.9999999999999998
    


```python
pred_test = predict(test_x, test_y, parameters)
```

    Accuracy: 0.74
    


```python
print_mislabeled_images(classes, test_x, test_y, pred_test)
```


    
![png](DNN_Cat_image_classification_files/DNN_Cat_image_classification_19_0.png)
    



```python

```
