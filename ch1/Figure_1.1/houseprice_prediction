import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Here, we generate the house price datapoints for convenience. 
# You can turn to Kaggle to download the true house price datapoints.
np.random.seed(0)  
X, y = make_regression(n_samples=50, n_features=1, noise=200, random_state=42, bias=400);
X = X * 1000 + 2500  

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y, facecolors='none', edgecolors='black', label='Data points')  
plt.plot(X, y_pred, color='black', label='Fitted line')  
plt.xlabel('square feet')
plt.ylabel('price (1000 dollars)')
# plt.title('Linear Regression Fit')
plt.grid(True)
plt.legend()
plt.show()
