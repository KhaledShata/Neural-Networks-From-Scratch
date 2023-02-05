#!/usr/bin/env python
# coding: utf-8

# In[2]:


class Term:
    def __init__(self, data, _children=(), _op='' , label=''):
        self.data = data
        self.grad = 0.0 #Intially every variable has no effect on the network output
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op 
        self.label = label
        
    def __repr__(self):
        return f"Term(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Term) else Term(other)
        out = Term(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        # Z = X + Y , Network Total Ouput  = L
        # Chain Rule 
        # DL    DL    DZ
        # -- =  -- *  --
        # DX    DZ    DX
        out._backward = _backward

        return out
    def __radd__(self, other): # other + self
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Term) else Term(other)
        out = Term(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other): # other * self
        return self * other
      
    def __pow__(self , other):
        assert isinstance(other , (int , float))
        out = Term(self.data ** other, (self, ) , label='^')
        def _backward():
            self.grad = other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        return out    
    
    def __neg__(self):
        return self*(-1)
    
    def __sub__(self , other):
        return self+(-other)
    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __truediv__(self , other):
        return self*(other**(-1))
    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def exp(self):
        out = Term(np.exp(self.data) , (self  ,) ,label = 'exp')
        def _backward():
            self.grad = out.data * out.grad
        out._backward = _backward
        return out
                                
    def tanh(self):
        x = self.data
        t = (np.exp(2*x) -1) / (np.exp(2*x)+1)
        out = Term(t ,{self ,}, label='tanh')
        def _backward():
            self.grad += (1- t**2) *out.grad
        out._backward = _backward
        return out
    
    def back_propagation(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()
        


# In[ ]:




