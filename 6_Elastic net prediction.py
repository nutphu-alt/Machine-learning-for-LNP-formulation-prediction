import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_excel("./dataset/dataset.xlsx")

# Drop rows with missing outputs
df = df.dropna(subset=["MFI", "%Positive cells", "%EE", "Size (nm)", "PDI"])

# Select features and target
FEATURES = ["pH", "Helper lipid (%)", "PEG lipid (%)", "Ionizable lipid (%)"]
TARGETS = ["MFI", "%Positive cells", "%EE", "Size (nm)", "PDI"]

X = df[FEATURES].values

# Range of lipid component to predict
ranges = {
    "pH": np.linspace(3, 5, 5),
    "Helper lipid (%)": np.linspace(12.5, 17.5, 11),
    "PEG lipid (%)": np.linspace(1, 2, 3),
    "Ionizable lipid (%)": np.linspace(37.5, 52.5, 31)
}

print(ranges)

def ends_in_0_or_5(x):
    """Return True if value ends in .0 or .5"""
    return np.isclose(x % 1, 0) or np.isclose(x % 1, 0.5)

# Create factorial design
all_combinations = list(itertools.product(*ranges.values()))
opt_df = pd.DataFrame(all_combinations, columns=ranges.keys())

# Filter combinations
filtered_df = opt_df[
    (opt_df["pH"].apply(ends_in_0_or_5)) &
    (opt_df["Helper lipid (%)"].apply(ends_in_0_or_5)) &
    (opt_df["PEG lipid (%)"].apply(ends_in_0_or_5)) &
    (opt_df["Ionizable lipid (%)"].apply(ends_in_0_or_5))
].reset_index(drop=True)

trained_models = {}
model_rmse = {}
cv = LeaveOneOut()

# Hyperparameters for optimization
en_param = {
    "regressor__Elastic Net__alpha": Real(0.0001, 1000),
    "regressor__Elastic Net__l1_ratio": Real(0, 1),
    "regressor__Elastic Net__tol": Real(0.00001, 0.001)
}

def log10_transform(y):
    return np.log1p(y)

def inv_log10_transform(y):
    return np.expm1(y)

# Model prediction
for target in TARGETS:
    y = df[target].values.ravel()
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
    en_opt_best_params = en_opt.best_params_
    y_preds = en_opt.predict(X_test)
    model_rmse[target] = (np.sqrt(mean_squared_error(y_test, y_preds)))

    print("--" * 70)

    en_coefs = en_opt.best_estimator_.regressor_.named_steps["Elastic Net"].coef_

    coefs = pd.DataFrame(
        en_coefs,
        columns=["Coefficients"],
        index=FEATURES
    )

    print(f"Coefficient for {TARGETS}")
    print(coefs)
    print(f"Best parameters for {target}: {en_opt_best_params}")
    print(f"RMSE for {target}: {model_rmse[target]}")

    best_model = en_opt.best_estimator_
    trained_models[target] = best_model

    filtered_df[f"Predicted {target}"] = best_model.predict(filtered_df[FEATURES])

top_results = (
    filtered_df.sort_values("Predicted MFI", ascending=False)
    .head(20)
    .reset_index(drop=True)
)

# Value to be compared
custom_values = {
    "pH": 5,
    "Helper lipid (%)": 17.5,
    "PEG lipid (%)": 1.5,
    "Ionizable lipid (%)": 42.5
}

custom_df = pd.DataFrame([custom_values]).reset_index(drop=True)

for target in TARGETS:
    best_model_for_custom_df = trained_models[target]

    custom_df[f"Predicted {target}"] = best_model_for_custom_df.predict(custom_df[FEATURES])

base_value = custom_df["Predicted MFI"].iloc[0]
print(f"Custom value predicted MFI: {base_value}")

# Define cutoff parameters (None = no limit)
MFI_MINIMUM = (float(base_value) + float(model_rmse["MFI"]))
SIZE_MAXIMUM = None        
PDI_MAXIMUM = None          
POS_MINIMUM = None         
EE_MINIMUM = None          

print(f"MFI min: {MFI_MINIMUM}")

# Apply only constraints that are not None
if SIZE_MAXIMUM is not None:
    top_results = top_results[top_results["Predicted Size (nm)"] < SIZE_MAXIMUM]

if MFI_MINIMUM is not None:
    top_results = top_results[top_results["Predicted MFI"] > MFI_MINIMUM]

if POS_MINIMUM is not None:
    top_results = top_results[top_results["Predicted %Positive cells"] > POS_MINIMUM]

if EE_MINIMUM is not None:
    top_results = top_results[top_results["Predicted %EE"] > EE_MINIMUM]

if PDI_MAXIMUM is not None:
    top_results = top_results[top_results["Predicted PDI"] < PDI_MAXIMUM]

print("Top 10 Predicted Formulations:")
print(top_results.head(10))

# Append to top results
top_results = pd.concat([top_results, custom_df], ignore_index=True)

# Save to CSV
top_results.to_csv("prediction.csv", index=False)

# Create label for plotting
top_results["Condition"] = top_results.apply(
    lambda r: f"pH={r['pH']:.1f}, Helper={r['Helper lipid (%)']:.1f}, "
              f"PEG={r['PEG lipid (%)']:.1f}, Ionizable={r['Ionizable lipid (%)']:.1f}",
    axis=1
)

# Plot top MFI values
plt.figure(figsize=(10, 6))
sns.barplot(data=(top_results.head(10)), x="Predicted MFI", y="Condition", palette="viridis")
plt.xlabel("Predicted MFI")
plt.ylabel("Formulation Parameters")
plt.title("Top 10 Formulations with Highest Predicted MFI")
plt.tight_layout()
plt.show()
