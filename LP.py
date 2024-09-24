
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate data
n, p = 2000, 20
sigma = 3
X = np.random.randn(n, p)
beta = np.random.randn(p)
eps = sigma * np.random.randn(n)
y = X@beta + eps
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
n_train, n_test = len(y_train), len(y_test)

# Fit the model
LR = LinearRegression(fit_intercept=True, positive=False)
LR.fit(X_train, y_train, sample_weight=np.ones(n_train))
LR.fit(X_train, y_train, sample_weight=None)

# Extract coefficients & compute R2
beta_hat, beta0_hat = LR.coef_, LR.intercept_
print("Relative error in beta = ", np.linalg.norm(beta_hat - beta)/np.linalg.norm(beta))
y_train_pred = LR.predict(X_train)
R2 = LR.score(X_train, y_train, sample_weight=None)
RSS = np.linalg.norm(y_train - y_train_pred)**2
TSS = np.linalg.norm(y_train - np.mean(y_train))**2 # We have R2 == 1-(RSS/TSS)

y_test_pred = LR.predict(X_test)
RSS = np.linalg.norm(y_test - y_test_pred)**2
TSS = np.linalg.norm(y_test - np.mean(y_train))**2
OSR2 = 1 - RSS/TSS
OSR2

