import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_excel("dataset/dataset.xlsx")

# Drop non-numeric or irrelevant columns (e.g., Run)
df = df.drop(columns=["Run", "non-lysed LNP", "Lysed LNP"])

# Define inputs and outputs
X = df[["pH", "Helper lipid (%)", "PEG lipid (%)", "Ionizable lipid (%)"]]  # input features
y = df["MFI"]

en_param = {
    "regressor__Elastic Net__alpha": Real(0.0001, 1000),
    "regressor__Elastic Net__l1_ratio": Real(0, 1),
    "regressor__Elastic Net__tol": Real(0.00001, 0.001)
}

def log10_transform(y):
    return np.log1p(y)

def inv_log10_transform(y):
    return np.expm1(y)

cv = LeaveOneOut()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

en_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("Elastic Net", ElasticNet(random_state=42))
])

model = TransformedTargetRegressor(
    regressor=en_pipe,
    func=log10_transform,
    inverse_func=inv_log10_transform
)

en_opt = BayesSearchCV(
    estimator=model,
    search_spaces=en_param,
    n_iter=50,  # Bayesian steps
    cv=cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)

en_opt.fit(X_train, y_train)
en_opt.best_params_

# After fitting
train_preds = en_opt.predict(X_train)
test_preds = en_opt.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)

print(f"Train RMSE: {train_rmse:.2f}, R²: {train_r2:.2f}")
print(f"Test RMSE: {test_rmse:.2f}, R²: {test_r2:.2f}")

# Compute learning curve
train_sizes, train_scores, test_scores = learning_curve(
    en_opt, X, y, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1, 25)
)

# Calculate mean and std
train_mean = (np.mean(train_scores, axis=1))
train_std = (np.std(train_scores, axis=1))
test_mean = (np.mean(test_scores, axis=1))
test_std = (np.std(test_scores, axis=1))

# Plot
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='green', label='Test score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='green')
plt.title("Learning Curve")
plt.xlabel("Number of sample in the training set")
plt.yticks(np.arange(-2, 1.5, 0.5))
plt.xticks(np.arange(0, 24, 4))
plt.ylabel("R² score")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()
