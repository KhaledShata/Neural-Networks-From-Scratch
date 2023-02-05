#!/usr/bin/env python
# coding: utf-8

# In[2]:


class Neuron:
    def __init__(self , nin):
        # Intializing Weights & Bias
        # nin is number of inputs of neruron
        self.w = [Term(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Term(random.uniform(-1,1))
    def __call__(self, x):
        # W X +B
        act = np.dot(self.w , x) + self.b
        return act.tanh()
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self , nin , nout):
        # Each Layer has (nout) neurons where each neuron has nin # of inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self ,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params =[]
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params

class MLP:
    # MLP has List of Layers  , 
    # Each Layer has list of Neurons  , 
    #Each Neuorn has its List of Weights & bias .
    def __init__(self , nin , nouts):
        # nin is number of inputs
        # nouts here is a list that holds size of every layer
        sz = [nin] + nouts
        self.layers = [Layer(sz[i] , sz[i+1]) for i in range(len(nouts))]
        # every layer takes input and produce output which is the input for the next layer
    
    def __call__(self , x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params =[]
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params
                

