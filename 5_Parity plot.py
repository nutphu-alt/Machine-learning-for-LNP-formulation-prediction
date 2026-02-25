import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from skopt.space import Real
from skopt import BayesSearchCV

# Load data
df = pd.read_excel("loaded data")

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

# Plot actual vs predicted for MFI
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_preds)
plt.xlabel("Actual MFI")
plt.ylabel("Predicted MFI")
plt.title(f"Actual vs Predicted MFI (R² = {r2_score(y_test, y_preds):.2f})")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()

#Residual analysis
residuals = y_preds - y_test
plt.scatter(y_preds, residuals, alpha=0.4)
plt.title('Residual Analysis MFI')
plt.show()
