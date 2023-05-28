import numpy as np
class Layer():
    def __init__(self, input_neuron, output_neuron, mean, std, bias_val, learning_rate, momentum, gamma, W_init):
        np.random.seed(42)
        # initialize learning rate
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.gamma = gamma #reqularized param
        # initialize weights with random normal numbers
        if W_init == 'He':
          self.weights = np.random.randn(input_neuron, output_neuron) / np.sqrt((input_neuron)/ 2)
        elif W_init == 'Zero':
          self.weights = np.full((input_neuron, output_neuron), 0)
        else:
          self.weights = np.random.normal(mean, std, (input_neuron, output_neuron))
          
        # initialize biases with zeros
        self.biases = np.full(output_neuron, bias_val)
        #for momentum
        self.weights_grad_prev = np.zeros((input_neuron, output_neuron))
        self.biases_grad_prev = np.zeros(output_neuron)
        
    def forward(self,input):
        return input @ self.weights + self.biases
    
    def backward(self,input,grad_out):
        grad_in = np.dot(grad_out,self.weights.T)
        weights_grad = np.transpose(np.dot(grad_out.T,input))
        biases_grad = np.sum(grad_out, axis = 0)
        # stochastic gradient descent step.
        self.weights_grad_prev = (self.momentum  * self.weights_grad_prev) + (self.learning_rate * (weights_grad - (self.gamma * self.weights)))
        self.weights = self.weights - self.weights_grad_prev
        self.biases_grad_prev = (self.momentum * self.biases_grad_prev) + (self.learning_rate * biases_grad )
        self.biases = self.biases - self.biases_grad_prev
        return grad_in

    def get_weigth(self):
      return self.weights

"""## Activation Funcs"""

# Sigmoid activation function
class Sigmoid():
    def __init__(self):
        pass
    
    def forward(self, z):
        return 1 / (1 + np.exp(-z))
    
    def backward(self, z, grad_out, _):
        return self.forward(z)*(1.0 - self.forward(z))

# ReLU activation function
class ReLu():
    def __init__(self):
        pass
    
    def forward(self, input):
        return np.maximum(0,input)
    
    def backward(self, input, grad_out):
        tmp = input > 0
        return grad_out * tmp

class LeakyReLu():
  def __init__(self):
    pass

  def forward(self, z):
    # s -- max(z,0.01*z) as a numpy array 
    s1 = z * (z > 0)
    s2 = 0.01 * z * (z <= 0)
    return s1 + s2 

  def backward(self, Z, dA):
    dZ = dA * np.where(Z > 0, 1, 0.01)

    return dZ

class Tanh():
  def __init__(self):
    pass
  
  def forward(self, input):
    return np.tanh(input)
    
  def backward(self, input, grad_out):
    return 1 - np.tanh(input) * np.tanh(input)
