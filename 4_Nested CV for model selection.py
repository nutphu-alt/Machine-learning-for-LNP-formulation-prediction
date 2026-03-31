import pandas as pd
import numpy as np
from xlsxwriter import Workbook
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RepeatedKFold, LeaveOneOut, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Load data
df = pd.read_excel("dataset/dataset.xlsx")

# Drop row with missing data
df = df.dropna(subset=["MFI", "%Positive cells", "%EE", "Size (nm)", "PDI"])

# Select features and target
FEATURES = ["pH", "Helper lipid (%)", "PEG lipid (%)", "Ionizable lipid (%)"]
TARGET = "MFI"

X = df[FEATURES].values
y = df[TARGET].values

# Function for log transform
def log_transform(y):
    return np.log1p(y)

# Function to revert log transform
def inv_log_transform(y):
    return np.expm1(y)

# Parameters for each model
en_param = {
    "regressor__Elastic Net__alpha": Real(0.0001, 1000),
    "regressor__Elastic Net__l1_ratio": Real(0, 1),
    "regressor__Elastic Net__tol": Real(0.00001, 0.001)
}

ridge_param = {
    "regressor__Ridge__alpha": Real(0.0001, 1000, prior="log-uniform")
}

lasso_param = {
    "regressor__Lasso__alpha": Real(0.0001, 1000)
}

svr_param = {
    'regressor__SVR__C': Real(1, 100),
    'regressor__SVR__gamma': Real(0.001, 10)
}

rf_param = {
    'regressor__Random Forest__n_estimators': Integer(10, 500),
    'regressor__Random Forest__max_depth': Integer(1, 100),
    'regressor__Random Forest__min_samples_split': Integer(2, 10),
    'regressor__Random Forest__min_samples_leaf': Integer(1, 10),
    'regressor__Random Forest__max_leaf_nodes': Integer(2, 100)
}

xgb_param = {
    'regressor__XGBoost__n_estimators': Integer(5, 20000),
    'regressor__XGBoost__max_depth': Integer(1, 50),
    'regressor__XGBoost__learning_rate': Real(0.00001, 0.1),
    'regressor__XGBoost__subsample': Real(0.1, 1.0),
    'regressor__XGBoost__colsample_bytree': Real(0.1, 1.0),
    'regressor__XGBoost__reg_alpha': Real(0.0, 5),
    'regressor__XGBoost__reg_lambda': Real(0.0, 5),
    'regressor__XGBoost__gamma': Integer(0, 100),
    'regressor__XGBoost__min_child_weight': Integer(0, 100),
    'regressor__XGBoost__max_delta_step': Integer(0, 100),
    'regressor__XGBoost__scale_pos_weight': Integer(1, 100)
}

mlp_param = {
    'regressor__MLP__hidden_layer_sizes': [(2,), (4,), (8,), (4,4), (4,2), (4,4,2)],
    'regressor__MLP__alpha': (0.0001, 100),
    'regressor__MLP__activation': ['relu', 'tanh'],
    'regressor__MLP__learning_rate_init': (0.00001, 0.01)
}

param_dict = {
    "Elastic Net": en_param,
    "Ridge": ridge_param,
    "Lasso": lasso_param,
    "SVR": svr_param,
    "Random Forest": rf_param,
    "XGBoost": xgb_param,
    "MLP": mlp_param
}

ML_model = [
    ("Elastic Net", ElasticNet(random_state=42)),
    ("Ridge", Ridge(random_state=42)),
    ("Lasso", Lasso(random_state=42)),
    ("SVR", SVR(max_iter=-1)),
    ("Random Forest", RandomForestRegressor(random_state=42)),
    ("XGBoost", XGBRegressor(random_state=42)),
    ("MLP", MLPRegressor(solver="lbfgs", max_iter=2000, random_state=42))
]

results_df_ = {}

# Path for saving output file
excel_path = "saved file"
writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

# Hyper parameter optimization and matices evaluation
for name, model_obj in ML_model:
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        (name, model_obj)
    ])

    model = TransformedTargetRegressor(
        regressor=base_model,
        func=log_transform,
        inverse_func=inv_log_transform
    )

    outer_cv = LeaveOneOut()
    inner_cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    outer_results = []
    all_y_true = []
    all_y_pred = []

    fold = 1
    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if name == "MLP":
            search = GridSearchCV(
                estimator=model,
                param_grid=param_dict[name],
                cv=inner_cv,
                scoring="neg_mean_squared_error",
                n_jobs=-1
            )
        else:
            search = BayesSearchCV(
                estimator=model,
                search_spaces=param_dict[name],
                n_iter=50,
                cv=inner_cv,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                random_state=42
            )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        all_y_true.append(y_test[0])
        all_y_pred.append(y_pred[0])

        outer_results.append({
            "Fold": fold,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE (%)": mape
        })

        fold += 1

    final_overall_r2 = r2_score(all_y_true, all_y_pred)

    results_df_[name] = pd.DataFrame(outer_results)

    print("--" * 70)
    print(name)

    print("\nNested CV Results:")
    print(results_df_[name])

    print("\nMean Performance:")
    print(results_df_[name].mean(numeric_only=True))

    print(f"R2 score for {name}: {final_overall_r2}")

    results_df_[name].to_excel(writer, sheet_name=name, index=False)

writer.close()
