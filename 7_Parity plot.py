import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from skopt.space import Real
from skopt import BayesSearchCV

# Load data
df = pd.read_excel("dataset/dataset.xlsx")

# Drop outputs with missing values
df = df.dropna(subset=["MFI", "%Positive cells", "%EE", "Size (nm)", "PDI"])

# Select features
FEATURES = ["pH", "Helper lipid (%)", "PEG lipid (%)", "Ionizable lipid (%)"]
TARGET = ["MFI"]

X = df[FEATURES].values
y = df[TARGET]
y = y.values.ravel()

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

# Split the data (80% train, 20% test)
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
    n_iter=50,
    cv=cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
    )

en_opt.fit(X_train, y_train)
best_model = en_opt.best_estimator_
y_preds = best_model.predict(X_test)
y_train_preds = best_model.predict(X_train)

# Plot actual vs predicted for training set MFI
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_train, y=y_train_preds, color='orange',label='Train')
plt.xlabel("Actual MFI")
plt.ylabel("Predicted MFI")
plt.title(f"Actual vs Predicted MFI of training set (R² = {r2_score(y_train, y_train_preds):.2f})")
plt.plot([y_train.min(), y_train.max()],
         [y_train.min(), y_train.max()], 'r--')
plt.grid(True)
plt.tight_layout()
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Plot actual vs predicted for test set MFI
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_preds, color='blue',label='Test')
plt.xlabel("Actual MFI")
plt.ylabel("Predicted MFI")
plt.title(f"Actual vs Predicted MFI of test set (R² = {r2_score(y_test, y_preds):.2f})")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.tight_layout()
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.show()

#Residual analysis
residuals = y_train_preds - y_train
plt.scatter(y_train_preds, residuals, alpha=0.4)
plt.title('Residual Analysis of training set MFI')
plt.yticks([-10000, -7500, -5000, -2500, 0, 2500, 5000, 7500, 10000], [r'-10,000', '-7,500', '-5,000', '-2,500', '0', '2,500', '5,000', '7,500', '10,000'])
plt.xticks([20000, 25000, 30000, 35000, 40000, 45000], [r'20,000', '25,000', '30,000', '35,000', '40,000', '45,000'])
plt.axhline(y=0, color='black', linestyle='--')
plt.show()

residuals = y_preds - y_test
plt.scatter(y_preds, residuals, alpha=0.4)
plt.title('Residual Analysis of test set MFI')
plt.ylim(-14000, 14000)
plt.yticks([-12500, -10000, -7500, -5000, -2500, 0, 2500, 5000, 7500, 10000, 12500], [r'-12,500', '-10,000', '-7,500', '-5,000', '-2,500', '0', '2,500', '5,000', '7,500', '10,000', '12,500'])
plt.xticks([25000, 30000, 35000, 40000, 45000, 50000], [r'25,000', '30,000', '35,000', '40,000', '45,000', '50,000'])
plt.axhline(y=0, color='black', linestyle='--')
plt.show()
