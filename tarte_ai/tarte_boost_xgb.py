"""Estimators for TARTE-Boost with XGBoost (compatible with sklearn interface)."""

import numpy as np
from copy import deepcopy
from typing import Union
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.preprocessing import label_binarize
from tarte_ai.gbdt_bagging_es import XGBRegressor_ESB, XGBClassifier_ESB
from tarte_ai.tarte_utils import reshape_pred_output


def _set_ridge_estimator(task):
    """Set ridge estimator depending on the task."""
    # Fixed params for ridge
    fixed_params_ridge = dict()
    fixed_params_ridge["alphas"] = [1e-2, 1e-1, 1, 10, 100]
    # Set estimator
    if task == "regression":
        return RidgeCV(**fixed_params_ridge)
    elif task == "classification":
        return RidgeClassifierCV(**fixed_params_ridge)


def _set_xgb_estimator(task, xgb_params={}):
    """Set xgboost estimator for boosting on the task."""
    if task == "regression":
        return XGBRegressor_ESB(**xgb_params)
    elif task == "classification":
        return XGBClassifier_ESB(**xgb_params)


def _calculate_ridge_output(estimator, X_test, task):
    """Calculate prediction based on the give task."""
    if task == "regression":
        y_pred = estimator.predict(X_test)
    elif task == "classification":
        y_pred = estimator.decision_function(X_test)
    return y_pred


def _calculate_xgb_output(estimator, X_test, task):
    """Calculate prediction based on the give task."""
    if task == "regression":
        y_pred = estimator.predict(X_test)
    elif task == "classification":
        y_pred = estimator.predict_proba(X_test)
    return y_pred


def _calculate_output(estimator, X, model_name, task):
    "Wrapping function for calculating output."
    if model_name == "ridge":
        out = _calculate_ridge_output(estimator, X, task)
    elif model_name == "xgb":
        out = _calculate_xgb_output(estimator, X, task)
    return reshape_pred_output(out)


def _run_base_model(X, y, model_name, task, xgb_params={}):
    """Simple run on the model and residual calculation on given data."""
    # Set estimator
    if model_name == "ridge":
        estimator = _set_ridge_estimator(task)
    elif model_name == "xgb":
        estimator = _set_xgb_estimator(task, xgb_params)
    estimator.fit(X, y)
    y_pred = _calculate_output(estimator, X, model_name, task)
    y_pred = reshape_pred_output(y_pred)
    if (task == "classification") and len(np.unique(y)) > 2:
        y_ = label_binarize(y, classes=np.unique(y))
    else:
        y_ = y
    residual = y_ - y_pred
    return estimator, residual


class BaseTARTEEnsembleXGB(BaseEstimator):
    """Base class for TARTE with simple boosting."""

    def __init__(
        self,
        *,
        model_names,
        fit_order,
        early_stopping_patience,
        num_model,
        val_size,
        n_estimators,
        max_depth,
        min_child_weight,
        subsample,
        learning_rate,
        colsample_bylevel,
        colsample_bytree,
        gamma,
        alpha,
        reg_lambda,
        random_state,
        n_jobs,
    ):
        self.model_names = model_names
        self.fit_order = fit_order
        self.early_stopping_patience = early_stopping_patience
        self.num_model = num_model
        self.val_size = val_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.learning_rate = learning_rate
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit the TARTE model.

        Parameters
        ----------
        X : The input samples.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
               Fitted estimator.
        """

        # Preliminary settings
        self.is_fitted_ = False
        self.X_ = X
        self.y_ = y
        self._set_specific_settings()

        # Set the boost estimator
        xgb_params = dict()
        xgb_params["early_stopping_patience"] = self.early_stopping_patience
        xgb_params["num_model"] = self.num_model
        xgb_params["val_size"] = self.val_size
        xgb_params["random_state"] = self.random_state
        xgb_params["n_jobs"] = self.n_jobs
        xgb_params["n_estimators"] = self.n_estimators
        xgb_params["max_depth"] = self.max_depth
        xgb_params["min_child_weight"] = self.min_child_weight
        xgb_params["subsample"] = self.subsample
        xgb_params["learning_rate"] = self.learning_rate
        xgb_params["colsample_bylevel"] = self.colsample_bylevel
        xgb_params["colsample_bytree"] = self.colsample_bytree
        xgb_params["gamma"] = self.gamma
        xgb_params["alpha"] = self.alpha
        xgb_params["reg_lambda"] = self.reg_lambda
        self.xgb_params_ = deepcopy(xgb_params)

        # Set the appropriate inputs
        X_base = [np.array([x[i] for x in X]) for i in range(len(X[0]))]
        if self.fit_order == "fixed":
            self.sort_idx_ = [x for x in range(len(X[0]))]
        else:
            self.sort_idx_ = self.fit_order

        # Base estimator
        base_estimator, y_residual = _run_base_model(
            X_base[self.sort_idx_[0]],
            y,
            self.model_names[self.sort_idx_[0]],
            self.task_,
            xgb_params,
        )

        # Boost estimator
        boost_estimator_list_ = []
        if self.ensemble_ == "boosting":
            for idx in self.sort_idx_[1:]:
                boost_estimator, y_residual = _run_base_model(
                    X_base[idx],
                    y_residual,
                    self.model_names[idx],
                    "regression",
                    xgb_params,
                )
                boost_estimator_list_.append(boost_estimator)
            self.model_list_ = [base_estimator] + boost_estimator_list_
            self.task_list_ = [self.task_] + ["regression"] * len(boost_estimator_list_)
        elif self.ensemble_ == "bagging":
            for idx in self.sort_idx_[1:]:
                boost_estimator, _ = _run_base_model(
                    X_base[idx],
                    y,
                    self.model_names[idx],
                    self.task_,
                    self.device,
                )
                boost_estimator_list_.append(boost_estimator)
            self.model_list_ = [base_estimator] + boost_estimator_list_
            self.task_list_ = [self.task_] * len(self.sort_idx_)

        self.is_fitted_ = True

        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        p : ndarray, shape (n_samples,) for binary classification or (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, "is_fitted_")

        # Set the appropriate inputs
        X_base = [np.array([x[i] for x in X]) for i in range(len(X[0]))]
        X_base = [X_base[idx] for idx in self.sort_idx_]
        model_name_sorted = [self.model_names[idx] for idx in self.sort_idx_]

        out = [
            _calculate_output(model, X, name, task)
            for (model, X, name, task) in zip(
                self.model_list_, X_base, model_name_sorted, self.task_list_
            )
        ]

        if self.ensemble_ == "boosting":
            out = np.sum(out, axis=0)
        if self.ensemble_ == "bagging":
            out = np.average(out, weights=self.weights_, axis=0)

        # Control for nulls in prediction
        if np.isnan(out).sum() > 0:
            mean_pred = np.mean(self.y_)
            out[np.isnan(out)] = mean_pred

        return out

    def predict(self, X):
        """Predict classes for X.

        Parameters
        ----------
        X : The input samples. (n_samples)

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted output/classes.
        """
        check_is_fitted(self, "is_fitted_")
        if self.task_ == "regression":
            return self.predict_proba(X)
        elif self.task_ == "classification":
            if self.classes_.shape[0] == 2:
                return np.round(self.predict_proba(X))
            elif self.classes_.shape[0] > 2:
                return np.argmax(self.predict_proba(X), axis=1)

    def decision_function(self, X):
        """Compute the decision function of X.

        Parameters
        ----------
        X : list of graph objects with size (n_samples)
            The input samples.

        Returns
        -------
        decision : ndarray, shape (n_samples,)
        """
        decision = self.predict_proba(X)
        return decision

    def _set_specific_settings(self):
        """Set specific parameters required for the task(Overridden)."""
        self.ensemble_ = None
        self.task_ = None
        self.classes_ = None
        return None


class TARTEBoostRegressor_XGB(RegressorMixin, BaseTARTEEnsembleXGB):
    """TARTE Ridge boosted with XGBoost for Regression tasks.

    Parameters
    ----------
    model_names : list, default=[],
        List of model names for each component of input.
    fit_order : {'fixed', 'validate}, default='fixed'
        The order to set the base and boosting estimator.
        If set to 'fixed', the first model is set as the base estimator.
        No effect for Bagging estimators.
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy for boosting estimator.
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    n_estimators : int, default=100
        Number of gradient boosted trees. Equivalent to number of boosting rounds.
    max_depth : int, default=6
        Maximum tree depth for base learners.
    min_child_weight : float, default=1.0
        Minimum sum of instance weight(hessian) needed in a child.
    subsample : float, default=1.0
        Subsample ratio of the training instance.
    learning_rate : float, default=0.3
        Boosting learning rate (xgb's “eta”)
    colsample_bylevel : float, default=1.0
        Subsample ratio of columns for each level.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    gamma : float, default=0.0
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    alpha : float, default=0.0
        L1 regularization term on weights (xgb's alpha).
    reg_lambda : float, default=1.0
        L2 regularization term on weights (xgb's lambda).
    random_state : int or None, default=0
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    """

    def __init__(
        self,
        *,
        model_names: list = [],
        fit_order: Union[str, list, tuple] = "fixed",
        early_stopping_patience: Union[None, int] = 20,
        num_model: int = 1,
        val_size: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        learning_rate: float = 0.3,
        colsample_bylevel: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: Union[None, int] = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            model_names=model_names,
            fit_order=fit_order,
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            learning_rate=learning_rate,
            colsample_bylevel=colsample_bylevel,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            alpha=alpha,
            reg_lambda=reg_lambda,
        )

    def _set_specific_settings(self):
        """Set specific parameters required for the task."""
        self.ensemble_ = "boosting"
        self.task_ = "regression"
        return None


class TARTEBoostClassifier_XGB(ClassifierMixin, BaseTARTEEnsembleXGB):
    """TARTE Ridge boosted with XGBoost for Classification tasks.

    Parameters
    ----------
    model_names : list, defautly=[],
        List of model names for each component of input.
    fit_order : {'fixed', 'validate}, default='fixed'
        The order to set the base and boosting estimator.
        If set to 'fixed', the first model is set as the base estimator.
        No effect for Bagging estimators.
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy for boosting estimator.
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    n_estimators : int, default=100
        Number of gradient boosted trees. Equivalent to number of boosting rounds.
    max_depth : int, default=6
        Maximum tree depth for base learners.
    min_child_weight : float, default=1.0
        Minimum sum of instance weight(hessian) needed in a child.
    subsample : float, default=1.0
        Subsample ratio of the training instance.
    learning_rate : float, default=0.3
        Boosting learning rate (xgb's “eta”)
    colsample_bylevel : float, default=1.0
        Subsample ratio of columns for each level.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    gamma : float, default=0.0
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    alpha : float, default=0.0
        L1 regularization term on weights (xgb's alpha).
    reg_lambda : float, default=1.0
        L2 regularization term on weights (xgb's lambda).
    random_state : int or None, default=None
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    """

    def __init__(
        self,
        *,
        model_names: list = [],
        fit_order: Union[str, list, tuple] = "fixed",
        early_stopping_patience: Union[None, int] = 20,
        num_model: int = 1,
        val_size: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        learning_rate: float = 0.3,
        colsample_bylevel: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: Union[None, int] = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            model_names=model_names,
            fit_order=fit_order,
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            learning_rate=learning_rate,
            colsample_bylevel=colsample_bylevel,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            alpha=alpha,
            reg_lambda=reg_lambda,
        )

    def _set_specific_settings(self):
        """Set specific parameters required for the task."""
        self.ensemble_ = "boosting"
        self.task_ = "classification"
        self.classes_ = np.unique(self.y_)
        return None


class TARTEBaggingRegressor_XGB(RegressorMixin, BaseTARTEEnsembleXGB):
    """TARTE Ridge boosted with XGBoost for Regression tasks.

    Parameters
    ----------
    model_names : list, defautly=[],
        List of model names for each component of input.
    fit_order : {'fixed', 'validate}, default='fixed'
        The order to set the base and boosting estimator.
        If set to 'fixed', the first model is set as the base estimator.
        No effect for Bagging estimators.
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy for boosting estimator.
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    n_estimators : int, default=100
        Number of gradient boosted trees. Equivalent to number of boosting rounds.
    max_depth : int, default=6
        Maximum tree depth for base learners.
    min_child_weight : float, default=1.0
        Minimum sum of instance weight(hessian) needed in a child.
    subsample : float, default=1.0
        Subsample ratio of the training instance.
    learning_rate : float, default=0.3
        Boosting learning rate (xgb's “eta”)
    colsample_bylevel : float, default=1.0
        Subsample ratio of columns for each level.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    gamma : float, default=0.0
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    alpha : float, default=0.0
        L1 regularization term on weights (xgb's alpha).
    reg_lambda : float, default=1.0
        L2 regularization term on weights (xgb's lambda).
    random_state : int or None, default=0
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    """

    def __init__(
        self,
        *,
        model_names: list = [],
        fit_order: Union[str, list, tuple] = "fixed",
        early_stopping_patience: Union[None, int] = 20,
        num_model: int = 1,
        val_size: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        learning_rate: float = 0.3,
        colsample_bylevel: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: Union[None, int] = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            model_names=model_names,
            fit_order=fit_order,
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            learning_rate=learning_rate,
            colsample_bylevel=colsample_bylevel,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            alpha=alpha,
            reg_lambda=reg_lambda,
        )

    def _set_specific_settings(self):
        """Set specific parameters required for the task."""
        self.ensemble_ = "bagging"
        self.task_ = "regression"
        return None


class TARTEBaggingClassifier_XGB(ClassifierMixin, BaseTARTEEnsembleXGB):
    """TARTE Ridge boosted with XGBoost for Classification tasks.

    Parameters
    ----------
    model_names : list, defautly=[],
        List of model names for each component of input.
    fit_order : {'fixed', 'validate}, default='fixed'
        The order to set the base and boosting estimator.
        If set to 'fixed', the first model is set as the base estimator.
        No effect for Bagging estimators.
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy for boosting estimator.
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    n_estimators : int, default=100
        Number of gradient boosted trees. Equivalent to number of boosting rounds.
    max_depth : int, default=6
        Maximum tree depth for base learners.
    min_child_weight : float, default=1.0
        Minimum sum of instance weight(hessian) needed in a child.
    subsample : float, default=1.0
        Subsample ratio of the training instance.
    learning_rate : float, default=0.3
        Boosting learning rate (xgb's “eta”)
    colsample_bylevel : float, default=1.0
        Subsample ratio of columns for each level.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    gamma : float, default=0.0
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    alpha : float, default=0.0
        L1 regularization term on weights (xgb's alpha).
    reg_lambda : float, default=1.0
        L2 regularization term on weights (xgb's lambda).
    random_state : int or None, default=None
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    """

    def __init__(
        self,
        *,
        model_names: list = [],
        fit_order: Union[str, list, tuple] = "fixed",
        early_stopping_patience: Union[None, int] = 20,
        num_model: int = 1,
        val_size: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        learning_rate: float = 0.3,
        colsample_bylevel: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: Union[None, int] = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            model_names=model_names,
            fit_order=fit_order,
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            learning_rate=learning_rate,
            colsample_bylevel=colsample_bylevel,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            alpha=alpha,
            reg_lambda=reg_lambda,
        )

    def _set_specific_settings(self):
        """Set specific parameters required for the task."""
        self.ensemble_ = "bagging"
        self.task_ = "classification"
        self.classes_ = np.unique(self.y_)
        return None
