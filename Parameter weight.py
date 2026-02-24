import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from skopt.space import Real
from skopt import BayesSearchCV

# Load data
df = pd.read_excel("saved data")

# Drop rows with missing outputs
df = df.dropna(subset=["MFI", "%Positive cells", "%EE", "Size (nm)", "PDI"])

# Select features and target
FEATURES = ["PEG lipid (%)", "Ionizable lipid (%)", "Helper lipid (%)", "pH"]
TARGET = "MFI"

X = df[FEATURES].values
y = df[TARGET]
y = y.values.ravel()

# Function for log transform
def log10_transform(y):
    return np.log1p(y)

# Function to revert log transform
def inv_log10_transform(y):
    return np.expm1(y)

en_param = {
    "regressor__Elastic Net__alpha": Real(0.0001, 1000),
    "regressor__Elastic Net__l1_ratio": Real(0, 1),
    "regressor__Elastic Net__tol": Real(0.00001, 0.001)
}

cv = LeaveOneOut()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("Elastic Net", ElasticNet(random_state=42))
])

model = TransformedTargetRegressor(
    regressor=model_pipeline,
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

best_model = en_opt.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

model_coefs = en_opt.best_estimator_.regressor_.named_steps["Elastic Net"].coef_

coefs = pd.DataFrame(
    model_coefs,
    columns=["Coefficients"],
    index=FEATURES
)

print(coefs)

fig, axes = plt.subplots(figsize=(12, 5))
axes.barh(coefs.index, coefs["Coefficients"], color="crimson")
axes.set_xlabel("Coefficient values")
axes.set_ylabel("Features")
axes.set_xlim(-0.2, 0.2)
axes.axvline(0, color="dimgray")
axes.set_title("Feature importance")
plt.tight_layout()
plt.show()
