import numpy as np
import matplotlib.pyplot as plt

# Funkcja sigmoidalna
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Gradient funkcji sigmoidalnej
def sigmoid_gradient(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

# Zakres danych x
x = np.linspace(-7, 7, 100)

# Obliczamy wartości funkcji sigmoidalnej i jej gradientu
sigmoid_values = sigmoid(x)
sigmoid_gradient_values = sigmoid_gradient(x)

# Tworzymy wykres
plt.figure(figsize=(8, 6))
plt.plot(x, sigmoid_values, label='Sigmoidalna')
plt.plot(x, sigmoid_gradient_values, label='Gradient Sigmoidalnej')
plt.legend()
plt.xlabel('x')
plt.ylabel('Wartosc')
plt.title('Funkcja Sigmoidalna + Gradient')
plt.grid(True)
plt.show()