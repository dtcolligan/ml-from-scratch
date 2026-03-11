print("Hello world")

import random

# The Value class - autograd engine
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._children = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f'**{k}')
        def _backward():
            self.grad += k * (self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-1 * other)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)
        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


# Neuron - stores weights and bias, computes w*x + b then relu
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b).tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


# Layer - a list of independent neurons
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


# MLP - a list of layers chained together
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# Training
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])

for i in range(100):
    # 1. Forward pass
    predictions = [model(x) for x in xs]

    # 2. Compute loss (MSE)
    loss = sum((pred - y) ** 2 for pred, y in zip(predictions, ys))

    # 3. Zero all gradients
    for p in model.parameters():
        p.grad = 0

    # 4. Backward pass
    loss.backward()

    # 5. Update parameters
    for p in model.parameters():
        p.data -= 0.1 * p.grad

    # Print loss every 10 steps
    if i % 10 == 0:
        print(f"Step {i}, Loss: {loss.data:.4f}")

# Final predictions
print("\nFinal predictions:")
for x, y in zip(xs, ys):
    pred = model(x)
    print(f"  Target: {y}, Predicted: {pred.data:.4f}")

import matplotlib.pyplot as plt

lr = 0.1 / (1 + 0.01 * i)

model = MLP(3, [4, 4, 1])
losses = []

for i in range(100):
    predictions = [model(x) for x in xs]
    loss = sum((pred - y) ** 2 for pred, y in zip(predictions, ys))
    losses.append(loss.data)

    for p in model.parameters():
        p.grad = 0
    loss.backward()
    for p in model.parameters():
        p.data -= lr * p.grad
# Loss curve
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# Final predictions vs targets
preds = [model(x).data for x in xs]
plt.figure(figsize=(6, 4))
plt.bar(range(4), ys, alpha=0.5, label='Target')
plt.bar(range(4), preds, alpha=0.5, label='Predicted')
plt.legend()
plt.title('Predictions vs Targets')
plt.show()
