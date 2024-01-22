import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('war1.csv', sep=';')
data['y'] = data['y'].str.replace(',', '.', regex=True).astype(float)
data['y'] = data['y'].astype(int)
x1 = data['x1'].values
x2 = data['x2'].values
y = data['y'].values
X = np.column_stack((x1, x2, np.ones_like(x1)))
X_pseudo_inv = np.linalg.pinv(X)
b = np.dot(X_pseudo_inv, y)
a, b = b[0], b[1]
x1_reg = np.linspace(min(x1), max(x1), 100)
x2_reg = np.linspace(min(x2), max(x2), 100)
y_reg = a * x1_reg + b * x2_reg
plt.scatter(x1, x2, c='blue', label='Dane')
plt.plot(x1_reg, x2_reg, c='red', label=f'Regresja: y = {a:.2f} * x1 + {b:.2f} * x2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()