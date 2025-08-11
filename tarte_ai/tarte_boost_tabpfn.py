"""Estimators for TARTE-Boost with TabPFN (compatible with sklearn interface)."""

import numpy as np
from typing import Union
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.preprocessing import label_binarize
from tabpfn import TabPFNRegressor, TabPFNClassifier
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


def _set_tabpfn_estimator(task, device):
    """Set xgboost estimator for boosting on the task."""

    if task == "regression":
        return TabPFNRegressor(device=device)
    elif task == "classification":
        return TabPFNClassifier(device=device)


def _calculate_ridge_output(estimator, X_test, task):
    """Calculate prediction based on the give task."""
    if task == "regression":
        y_pred = estimator.predict(X_test)
    elif task == "classification":
        y_pred = estimator.decision_function(X_test)
    return y_pred


def _calculate_tabpfn_output(estimator, X_test, batch_size, task):
    """Calculate tabpfn output for large datasets."""

    # Check with tabpfn repo for more efficient calculation of the output
    test_size = len(X_test)
    if test_size < batch_size:
        if task == "regression":
            y_pred = estimator.predict(X_test)
        else:
            y_pred = estimator.predict_proba(X_test)
    else:
        mok = test_size // batch_size
        if task == "regression":
            y_pred = np.empty(shape=(0,))
        else:
            y_pred = np.empty(shape=(0, 2))
        for x in range(mok):
            idx_1 = x * batch_size
            idx_2 = (x + 1) * batch_size
            if task == "regression":
                y_pred_ = estimator.predict(X_test[idx_1:idx_2])
                y_pred = np.hstack([y_pred, y_pred_])
            else:
                y_pred_ = estimator.predict_proba(X_test[idx_1:idx_2])
                y_pred = np.vstack([y_pred, y_pred_])
        if task == "regression":
            y_pred_ = estimator.predict(X_test[idx_2:])
            y_pred = np.hstack([y_pred, y_pred_])
        else:
            y_pred_ = estimator.predict_proba(X_test[idx_2:])
            y_pred = np.vstack([y_pred, y_pred_])
    return y_pred


def _calculate_output(estimator, X, model_name, task):
    "Wrapping function for calculating output."
    if model_name == "ridge":
        out = _calculate_ridge_output(estimator, X, task)
    elif model_name == "tabpfn":
        out = _calculate_tabpfn_output(estimator, X, 8192, task)
    return reshape_pred_output(out)


def _run_base_model(X, y, model_name, task, device):
    """Simple run on the model and residual calculation on given data."""
    # Set estimator
    if model_name == "ridge":
        estimator = _set_ridge_estimator(task)
        estimator.fit(X, y)
        y_pred = _calculate_ridge_output(estimator, X, task)
    elif model_name == "tabpfn":
        estimator = _set_tabpfn_estimator(task, device)
        estimator.fit(X, y)
        y_pred = _calculate_tabpfn_output(estimator, X, 8192, task)
    y_pred = reshape_pred_output(y_pred)
    if (task == "classification") and len(np.unique(y)) > 2:
        y_ = label_binarize(y, classes=np.unique(y))
    else:
        y_ = y
    residual = y_ - y_pred
    return estimator, residual


class BaseTARTEEnsembleTabPFN(BaseEstimator):
    """Base class for TARTE with simple boosting."""

    def __init__(
        self,
        *,
        model_names,
        fit_order,
        device,
    ):
        self.model_names = model_names
        self.fit_order = fit_order
        self.device = device

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
            self.device,
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
                    self.device,
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
            out = np.average(out, axis=0)

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


class TARTEBoostRegressor_TabPFN(RegressorMixin, BaseTARTEEnsembleTabPFN):
    """TARTE Ridge boosted with XGBoost for Regression tasks.

    Parameters
    ----------
    model_names : list, default=[],
        List of model names for each component of input.
    fit_order : {'fixed', 'train_score}, default='fixed'
        The order to set the base and boosting estimator.
        If set to 'fixed', the first model is set as the base estimator.
        No effect for Bagging estimators.
    device : {"cpu", "cuda"}, default="cuda",
        The device used for the estimator.
    """

    def __init__(
        self,
        *,
        model_names: list = [],
        fit_order: Union[str, list, tuple] = "fixed",
        device: str = "cpu",
    ):
        super().__init__(
            model_names=model_names,
            fit_order=fit_order,
            device=device,
        )

    def _set_specific_settings(self):
        """Set specific parameters required for the task."""
        self.ensemble_ = "boosting"
        self.task_ = "regression"
        return None


class TARTEBoostClassifier_TabPFN(ClassifierMixin, BaseTARTEEnsembleTabPFN):
    """TabPFN boosted with TARTE Ridge  for Classification tasks.

    Parameters
    ----------
    model_names : list, defautly=[],
        List of model names for each component of input.
    fit_order : {'fixed', 'train_score}, default='fixed'
        The order to set the base and boosting estimator.
        If set to 'fixed', the first model is set as the base estimator.
        No effect for Bagging estimators.
    device : {"cpu", "cuda"}, default="cuda",
        The device used for the estimator.
    """

    def __init__(
        self,
        *,
        model_names: list = [],
        fit_order: Union[str, list, tuple] = "fixed",
        device: str = "cpu",
    ):
        super().__init__(
            model_names=model_names,
            fit_order=fit_order,
            device=device,
        )

    def _set_specific_settings(self):
        """Set specific parameters required for the task."""
        self.ensemble_ = "boosting"
        self.task_ = "classification"
        self.classes_ = np.unique(self.y_)
        return None


class TARTEBaggingRegressor_TabPFN(RegressorMixin, BaseTARTEEnsembleTabPFN):
    """TARTE Ridge boosted with XGBoost for Regression tasks.

    Parameters
    ----------
    model_names : list, defautly=[],
        List of model names for each component of input.
    fit_order : {'fixed', 'train_score}, default='fixed'
        The order to set the base and boosting estimator.
        If set to 'fixed', the first model is set as the base estimator.
        No effect for Bagging estimators.
    device : {"cpu", "cuda"}, default="cuda",
        The device used for the estimator.
    """

    def __init__(
        self,
        *,
        model_names: list = [],
        fit_order: Union[str, list, tuple] = "fixed",
        device: str = "cpu",
    ):
        super().__init__(
            model_names=model_names,
            fit_order=fit_order,
            device=device,
        )

    def _set_specific_settings(self):
        """Set specific parameters required for the task."""
        self.ensemble_ = "bagging"
        self.task_ = "regression"
        return None


class TARTEBaggingClassifier_TabPFN(ClassifierMixin, BaseTARTEEnsembleTabPFN):
    """TabPFN boosted with TARTE Ridge  for Classification tasks.

    Parameters
    ----------
    model_names : list, defautly=[],
        List of model names for each component of input.
    fit_order : {'fixed', 'train_score}, default='fixed'
        The order to set the base and boosting estimator.
        If set to 'fixed', the first model is set as the base estimator.
        No effect for Bagging estimators.
    device : {"cpu", "cuda"}, default="cuda",
        The device used for the estimator.
    """

    def __init__(
        self,
        *,
        model_names: list = [],
        fit_order: Union[str, list, tuple] = "fixed",
        device: str = "cpu",
    ):
        super().__init__(
            model_names=model_names,
            fit_order=fit_order,
            device=device,
        )

    def _set_specific_settings(self):
        """Set specific parameters required for the task."""
        self.ensemble_ = "bagging"
        self.task_ = "classification"
        self.classes_ = np.unique(self.y_)
        return None
