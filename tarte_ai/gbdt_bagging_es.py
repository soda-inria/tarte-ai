"""Simple estimators for XGBoost/CatBoost with bagging/early-stopping (compatible with sklearn interface)."""

import numpy as np
from typing import Union
from copy import deepcopy
from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from xgboost import callback, XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from tarte_ai.tarte_utils import reshape_pred_output


def _set_train_valid_split(X, y, n_splits, val_size, random_state, task):
    """Train/validation split for the bagging strategy.

    The style of split depends on the cross_validate parameter.
    Reuturns the train/validation split with KFold cross-validation.
    """
    if task == "regression":
        splitter = ShuffleSplit(
            n_splits=n_splits,
            test_size=val_size,
            random_state=random_state,
        )
        splits = [
            (train_index, test_index)
            for train_index, test_index in splitter.split(np.arange(0, len(X)))
        ]
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=val_size,
            random_state=random_state,
        )
        splits = [
            (train_index, test_index)
            for train_index, test_index in splitter.split(np.arange(0, len(X)), y)
        ]
    return splits


def _run_train(X, y, split_index, estimator):
    """Train each model corresponding to the random_state.

    It sets train/valid set for the early stopping criterion only for XGB/HGB.
    Returns the trained model.
    """
    # Set train/validation for early stopping
    X_train, X_valid = X[split_index[0]], X[split_index[1]]
    y_train, y_valid = y[split_index[0]], y[split_index[1]]
    eval_set = [(X_valid, y_valid)]
    # Fit the model
    estimator_ = deepcopy(estimator)
    estimator_.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return estimator_


def _generate_output(X, y_train, model_list, task):
    """Generate the output from the trained model.

    Returns the output (prediction) of input X.
    """

    if task == "regression":
        out = [model.predict(X) for model in model_list]
    elif task == "classification":
        out = [model.predict_proba(X) for model in model_list]
    out = np.average(out, weights=None, axis=0)
    out = reshape_pred_output(out)

    # Control for nulls in prediction
    if np.isnan(out).sum() > 0:
        mean_pred = np.mean(y_train)
        out[np.isnan(out)] = mean_pred

    return out


class BASEGBDT_ESBagging(BaseEstimator):
    """Base class for GBDT Estimator with early-stopping and bagging."""

    def __init__(
        self,
        *,
        early_stopping_patience,
        num_model,
        val_size,
        random_state,
        n_jobs,
    ):
        super().__init__()

        self.early_stopping_patience = early_stopping_patience
        self.num_model = num_model
        self.val_size = val_size
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit the TARTE model.

        Parameters
        ----------
        X : pandas DataFrame or array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
               Fitted estimator.
        """

        # Preliminaries
        self.is_fitted_ = False
        self.X_ = X
        self.y_ = y
        self._set_specific_settings()

        # Set splits for train/vaidation
        splits = _set_train_valid_split(
            X,
            y,
            self.num_model,
            self.val_size,
            self.random_state,
            self.task_,
        )
        estimator = self._set_gbdt_estimator()

        # Fit model and store the required results that may be used later
        self.model_list_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_run_train)(X, y, split_index, estimator) for split_index in splits
        )
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

        return _generate_output(X, self.y_, self.model_list_, self.task_)

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
        """Set specific parameters required for the task."""
        if self._estimator_type == "regressor":
            self.task_ = "regression"
        elif self._estimator_type == "classifier":
            self.task_ = "classification"
            self.classes_ = np.unique(self.y_)
        return None

    def _set_gbdt_estimator():
        """Function to set the estimator(Overidden)."""
        return None


class XGBRegressor_ESB(RegressorMixin, BASEGBDT_ESBagging):
    """XGBoost Regressor (with early-stopping and bagging) for Regression tasks.

    Parameters
    ----------
    early_stopping_patience : int or None, default=50
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    random_state : int or None, default=None
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized over the number of models.
    n_estimators : int, default=1000
        Number of gradient boosted trees. Equivalent to number of boosting rounds.
    max_depth : int, default=6
        Maximum tree depth for base learners.
    min_child_weight : float, default=1.0
        Minimum sum of instance weight(hessian) needed in a child.
    subsample : float, default=1.0
        Subsample ratio of the training instance.
    learning_rate : float, default=0.3
        Boosting learning rate (xgb's "eta")
    colsample_bylevel : float, default=1.0
        Subsample ratio of columns for each level.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    gamma : float, default=0.0
        (min_split_loss) Minimum loss reduction required to make a further partition on a leaf node of the tree.
    alpha : float, default=0.0
        L1 regularization term on weights (xgb's alpha).
    reg_lambda : float, default=1.0
        L2 regularization term on weights (xgb's lambda).
    """

    def __init__(
        self,
        *,
        early_stopping_patience: Union[None, int] = 50,
        num_model: int = 1,
        val_size: float = 0.1,
        random_state: Union[None, int] = None,
        n_jobs: int = 1,
        n_estimators: int = 1000,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        learning_rate: float = 0.3,
        colsample_bylevel: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        alpha: float = 0.0,
        reg_lambda: float = 1.0,
    ):

        super().__init__(
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
        )

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

    def _set_gbdt_estimator(self):
        """Function to set the estimator with XGBoost parameters."""

        # Set estimator params
        xgb_params = dict()
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

        # Set Early-Stopping and return estimator
        if self.early_stopping_patience is not None:
            callbacks = [
                callback.EarlyStopping(
                    rounds=self.early_stopping_patience,
                    min_delta=1e-3,
                    save_best=True,
                    maximize=False,
                )
            ]
        else:
            callbacks = []
        return XGBRegressor(**xgb_params, callbacks=callbacks)


class XGBClassifier_ESB(ClassifierMixin, BASEGBDT_ESBagging):
    """XGBoost Classifier (with early-stopping and bagging) for Classification tasks.

    Parameters
    ----------
    early_stopping_patience : int or None, default=50
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    random_state : int or None, default=None
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized over the number of models.
    n_estimators : int, default=1000
        Number of gradient boosted trees. Equivalent to number of boosting rounds.
    max_depth : int, default=6
        Maximum tree depth for base learners.
    min_child_weight : float, default=1.0
        Minimum sum of instance weight(hessian) needed in a child.
    subsample : float, default=1.0
        Subsample ratio of the training instance.
    learning_rate : float, default=0.3
        Boosting learning rate (xgb's "eta")
    colsample_bylevel : float, default=1.0
        Subsample ratio of columns for each level.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    gamma : float, default=0.0
        (min_split_loss) Minimum loss reduction required to make a further partition on a leaf node of the tree.
    alpha : float, default=0.0
        L1 regularization term on weights (xgb's alpha).
    reg_lambda : float, default=1.0
        L2 regularization term on weights (xgb's lambda).
    """

    def __init__(
        self,
        *,
        early_stopping_patience: Union[None, int] = 50,
        num_model: int = 1,
        val_size: float = 0.1,
        random_state: Union[None, int] = None,
        n_jobs: int = 1,
        n_estimators: int = 1000,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        learning_rate: float = 0.3,
        colsample_bylevel: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        alpha: float = 0.0,
        reg_lambda: float = 1.0,
    ):

        super().__init__(
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
        )

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

    def _set_gbdt_estimator(self):
        """Function to set the estimator with XGBoost parameters."""

        # Set estimator params
        xgb_params = dict()
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

        # Set Early-Stopping and return estimator
        if self.early_stopping_patience is not None:
            callbacks = [
                callback.EarlyStopping(
                    rounds=self.early_stopping_patience,
                    min_delta=1e-3,
                    save_best=True,
                    maximize=False,
                )
            ]
        else:
            callbacks = []
        return XGBClassifier(**xgb_params, callbacks=callbacks)


class CatBoostRegressor_ESB(RegressorMixin, BASEGBDT_ESBagging):
    """CatBoost Regressor (with early-stopping and bagging) for Regression tasks.

    Parameters
    ----------
    early_stopping_patience : int or None, default=50
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    random_state : int or None, default=None
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized over the number of models.
    thread_count : int, default=1
        The number of threads to use during the training.
    cat_features : list or None, default=None
        A one-dimensional array of categorical columns indices.
    iterations : int, default=1000
        The maximum number of trees that can be built when solving machine learning problems.
    boosting_type : {'Ordered', 'Plain'}, default='Plain'
        Boosting scheme.
    max_depth : int, default=6
        Depth of the trees.
    learning_rate : float, default=0.3
        The learning rate. Used for reducing the gradient step.
    bagging_temperature : float, default=1.0
        Defines the settings of the Bayesian bootstrap. 
        It is used by default in classification and regression modes.
    l2_leaf_reg : float, default=3.0
        Coefficient at the L2 regularization term of the cost function.
    random_strength : float, default=1.0
        The amount of randomness to use for scoring splits when the tree structure is selected. 
        Use this parameter to avoid overfitting the model.
    one_hot_max_size : int, default=2
        Use one-hot encoding for all categorical features with a number of different values less than or equal to the given parameter value. 
        Ctrs are not calculated for such features.
    leaf_estimation_iterations : int or None, default=None
        CatBoost might calculate leaf values using several gradient or newton steps instead of a single one.
        This parameter regulates how many steps are done in every tree when calculating leaf values.
    """

    def __init__(
        self,
        *,
        early_stopping_patience: Union[None, int] = 50,
        num_model: int = 1,
        val_size: float = 0.1,
        random_state: Union[None, int] = None,
        n_jobs: int = 1,
        thread_count: int = 1,
        cat_features: Union[None, list] = None,
        iterations: int = 1000,
        boosting_type: str = "Plain",
        max_depth: int = 6,
        learning_rate: float = 0.3,
        bagging_temperature: float = 1.0,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        one_hot_max_size: int = 2,
        leaf_estimation_iterations: Union[None, int] = None,
    ):

        super().__init__(
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.thread_count = thread_count
        self.cat_features = cat_features
        self.iterations = iterations
        self.boosting_type = boosting_type
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.bagging_temperature = bagging_temperature
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.one_hot_max_size = one_hot_max_size
        self.leaf_estimation_iterations = leaf_estimation_iterations

    def _set_gbdt_estimator(self):
        """Function to set the estimator with CatBoost parameters."""

        # Set estimator params
        cb_params = dict()
        cb_params["allow_writing_files"] = False
        cb_params["thread_count"] = self.thread_count
        cb_params["cat_features"] = self.cat_features
        cb_params["iterations"] = self.iterations
        cb_params["boosting_type"] = self.boosting_type
        cb_params["max_depth"] = self.max_depth
        cb_params["learning_rate"] = self.learning_rate
        cb_params["bagging_temperature"] = self.bagging_temperature
        cb_params["l2_leaf_reg"] = self.l2_leaf_reg
        cb_params["random_strength"] = self.random_strength
        cb_params["one_hot_max_size"] = self.one_hot_max_size
        cb_params["leaf_estimation_iterations"] = self.leaf_estimation_iterations
        if self.early_stopping_patience is not None:
            cb_params["od_type"] = "Iter"
            cb_params["od_wait"] = self.early_stopping_patience
        self.cb_params_ = deepcopy(cb_params)

        return CatBoostRegressor(**cb_params)


class CatBoostClassifier_ESB(ClassifierMixin, BASEGBDT_ESBagging):
    """CatBoost Classifier (with early-stopping and bagging) for Classification tasks.

    Parameters
    ----------
    early_stopping_patience : int or None, default=50
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    num_model : int, default=1
        The total number of models used for Bagging strategy
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    random_state : int or None, default=None
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized over the number of models.
    thread_count : int, default=1
        The number of threads to use during the training.
    cat_features : list or None, default=None
        A one-dimensional array of categorical columns indices.
    iterations : int, default=1000
        The maximum number of trees that can be built when solving machine learning problems.
    boosting_type : {'Ordered', 'Plain'}, default='Plain'
        Boosting scheme.
    max_depth : int, default=6
        Depth of the trees.
    learning_rate : float, default=0.3
        The learning rate. Used for reducing the gradient step.
    bagging_temperature : float, default=1.0
        Defines the settings of the Bayesian bootstrap. 
        It is used by default in classification and regression modes.
    l2_leaf_reg : float, default=3.0
        Coefficient at the L2 regularization term of the cost function.
    random_strength : float, default=1.0
        The amount of randomness to use for scoring splits when the tree structure is selected. 
        Use this parameter to avoid overfitting the model.
    one_hot_max_size : int, default=2
        Use one-hot encoding for all categorical features with a number of different values less than or equal to the given parameter value. 
        Ctrs are not calculated for such features.
    leaf_estimation_iterations : int or None, default=None
        CatBoost might calculate leaf values using several gradient or newton steps instead of a single one.
        This parameter regulates how many steps are done in every tree when calculating leaf values.
    """

    def __init__(
        self,
        *,
        early_stopping_patience: Union[None, int] = 50,
        num_model: int = 1,
        val_size: float = 0.1,
        random_state: Union[None, int] = None,
        n_jobs: int = 1,
        thread_count: int = 1,
        cat_features: Union[None, list] = None,
        iterations: int = 1000,
        boosting_type: str = "Plain",
        max_depth: int = 6,
        learning_rate: float = 0.3,
        bagging_temperature: float = 1.0,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        one_hot_max_size: int = 2,
        leaf_estimation_iterations: Union[None, int] = None,
    ):

        super().__init__(
            early_stopping_patience=early_stopping_patience,
            num_model=num_model,
            val_size=val_size,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.thread_count = thread_count
        self.cat_features = cat_features
        self.iterations = iterations
        self.boosting_type = boosting_type
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.bagging_temperature = bagging_temperature
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.one_hot_max_size = one_hot_max_size
        self.leaf_estimation_iterations = leaf_estimation_iterations

    def _set_gbdt_estimator(self):
        """Function to set the estimator with CatBoost parameters."""

        # Set estimator params
        cb_params = dict()
        cb_params["allow_writing_files"] = False
        cb_params["thread_count"] = self.thread_count
        cb_params["cat_features"] = self.cat_features
        cb_params["iterations"] = self.iterations
        cb_params["boosting_type"] = self.boosting_type
        cb_params["max_depth"] = self.max_depth
        cb_params["learning_rate"] = self.learning_rate
        cb_params["bagging_temperature"] = self.bagging_temperature
        cb_params["l2_leaf_reg"] = self.l2_leaf_reg
        cb_params["random_strength"] = self.random_strength
        cb_params["one_hot_max_size"] = self.one_hot_max_size
        cb_params["leaf_estimation_iterations"] = self.leaf_estimation_iterations
        if self.early_stopping_patience is not None:
            cb_params["od_type"] = "Iter"
            cb_params["od_wait"] = self.early_stopping_patience
        self.cb_params_ = deepcopy(cb_params)

        return CatBoostClassifier(**cb_params)
