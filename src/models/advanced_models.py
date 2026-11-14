"""
EstateIQ AI - Phase 5: Advanced Gradient Boosting Pipelines
------------------------------------------------------------
Implements production-grade regression pipelines for LightGBM, XGBoost, and CatBoost:
1. Model-specific preprocessing pipelines:
   - CatBoost: Native categorical handling with median & constant imputation.
   - LightGBM: Numerical median imputation, categorical constant imputation + sparse OneHotEncoder.
   - XGBoost: Numerical median imputation (no scaling), categorical constant imputation + sparse OneHotEncoder.
2. Target transformation helper:
   - TransformedTargetRegressor with np.log1p and np.expm1.
3. Custom CatBoost adapter supporting Scikit-Learn cloning, KFold cross-validation,
   and TransformedTargetRegressor compatibility.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Global Feature Schema
NUMERICAL_FEATURES = ["total_sqft", "bhk", "bath", "balcony"]
CATEGORICAL_FEATURES = ["location", "area_type"]
FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "price"
LEAKAGE_COLS = ["price_per_sqft"]

# Default Reproducible Hyperparameters
LGBM_CONFIG: Dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

XGB_CONFIG: Dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,    # L1 regularization to combat collinearity
    "reg_lambda": 1.0,   # L2 regularization
    "random_state": 42,
    "n_jobs": -1
}

CATBOOST_CONFIG: Dict[str, Any] = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": 0
}

RF_CONFIG: Dict[str, Any] = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1
}


class CatBoostNativeRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-Learn compliant wrapper for CatBoostRegressor.
    
    Prevents runtime clone issues with CatBoost's constructor mutation of cat_features
    while ensuring full compatibility with TransformedTargetRegressor and cross_val_score.
    """
    def __init__(
        self,
        iterations: int = 300,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        verbose: int = 0,
        cat_features: Optional[List[str]] = None
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.verbose = verbose
        self.cat_features = cat_features

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Determine categorical columns if not explicitly provided
        if self.cat_features is not None:
            cats = self.cat_features
        elif isinstance(X, pd.DataFrame):
            cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            cats = None

        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=self.verbose
        )
        self.model_.fit(X, y, cat_features=cats)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model_.get_feature_importance()


def build_tree_preprocessor_ohe(
    num_features: List[str] = NUMERICAL_FEATURES,
    cat_features: List[str] = CATEGORICAL_FEATURES,
    sparse_output: bool = True
) -> ColumnTransformer:
    """
    Builds an imputation + One-Hot Encoding ColumnTransformer for tree-based models (XGBoost, LightGBM, RF).
    Numerical features: Median Imputation (no unnecessary scaling).
    Categorical features: Constant 'missing' Imputation + OneHotEncoder(handle_unknown='ignore').
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features)
        ],
        remainder="drop"
    )
    return preprocessor


def build_tree_preprocessor_ordinal(
    num_features: List[str] = NUMERICAL_FEATURES,
    cat_features: List[str] = CATEGORICAL_FEATURES
) -> ColumnTransformer:
    """
    Builds an imputation + Ordinal Encoding ColumnTransformer for LightGBM histogram binning.
    Numerical features: Median Imputation.
    Categorical features: Constant 'missing' Imputation + OrdinalEncoder(handle_unknown='use_encoded_value').
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features)
        ],
        remainder="drop"
    )
    return preprocessor


def build_catboost_preprocessor(
    num_features: List[str] = NUMERICAL_FEATURES,
    cat_features: List[str] = CATEGORICAL_FEATURES
) -> ColumnTransformer:
    """
    Builds an imputation ColumnTransformer for CatBoost preserving native categorical columns.
    Outputs a pandas DataFrame with preserved column names and dtypes.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_pipeline(
    model_name: str,
    use_log_target: bool = False,
    categorical_encoding: str = "onehot",
    custom_params: Optional[Dict[str, Any]] = None
) -> Pipeline | TransformedTargetRegressor:
    """
    Factory function to build an end-to-end regression pipeline.

    Args:
        model_name: One of 'lightgbm', 'xgboost', 'catboost', 'random_forest'.
        use_log_target: If True, wraps model in TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1).
        categorical_encoding: For tree models with encoding options ('onehot' or 'ordinal').
        custom_params: Optional dictionary of hyperparameter overrides.

    Returns:
        Scikit-Learn Pipeline or TransformedTargetRegressor.
    """
    model_key = model_name.lower().strip()
    c_params = dict(custom_params or {})

    if model_key == "lightgbm":
        if "iterations" in c_params:
            c_params["n_estimators"] = c_params.pop("iterations")
        params = {**LGBM_CONFIG, **c_params}
        if categorical_encoding.lower() == "ordinal":
            preprocessor = build_tree_preprocessor_ordinal()
        else:
            preprocessor = build_tree_preprocessor_ohe(sparse_output=True)
        regressor = LGBMRegressor(**params)
        base_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

    elif model_key == "xgboost":
        if "iterations" in c_params:
            c_params["n_estimators"] = c_params.pop("iterations")
        params = {**XGB_CONFIG, **c_params}
        preprocessor = build_tree_preprocessor_ohe(sparse_output=True)
        regressor = XGBRegressor(**params)
        base_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

    elif model_key == "catboost":
        if "n_estimators" in c_params:
            c_params["iterations"] = c_params.pop("n_estimators")
        params = {**CATBOOST_CONFIG, **c_params}
        preprocessor = build_catboost_preprocessor()
        regressor = CatBoostNativeRegressor(
            cat_features=CATEGORICAL_FEATURES,
            **params
        )
        base_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

    elif model_key in ["random_forest", "rf"]:
        if "iterations" in c_params:
            c_params["n_estimators"] = c_params.pop("iterations")
        params = {**RF_CONFIG, **c_params}
        preprocessor = build_tree_preprocessor_ohe(sparse_output=True)
        regressor = RandomForestRegressor(**params)
        base_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

    else:
        raise ValueError(f"Unsupported model_name '{model_name}'. Choose from 'lightgbm', 'xgboost', 'catboost', 'random_forest'.")

    if use_log_target:
        return TransformedTargetRegressor(
            regressor=base_pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )
    return base_pipeline
