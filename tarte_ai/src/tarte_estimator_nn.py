"""Neural network compatible with sklearn."""

import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Union
from torch import Tensor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from joblib import Parallel, delayed


from torcheval.metrics import (
    MeanSquaredError,
    R2Score,
    BinaryAUROC,
    BinaryNormalizedEntropy,
    BinaryAUPRC,
    MulticlassAUROC,
)
from scipy.special import softmax
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedShuffleSplit,
)


## Simple MLP model
class MLP_Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        num_layers: int,
    ):
        super().__init__()

        self.initial = nn.Linear(input_dim, hidden_dim)

        self.mlp_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.Sequential(*[self.mlp_block for _ in range(num_layers)])

        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        X = self.initial(X)
        X = self.layers(X)
        X = self.classifier(X)
        return X


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


def _run_train_with_early_stopping(X, y, split_index, model, train_args):
    """Train each model corresponding to the random_state.

    It sets train/valid set for the early stopping criterion only for XGB/HGB.
    Returns the trained model.
    """

    # Set datasets
    ds_train_ = X[split_index[0], :]
    y_train_ = y[split_index[0]]
    ds_train = TensorDataset(ds_train_, y_train_)
    ds_valid_ = X[split_index[1], :]
    y_valid_ = y[split_index[1]]
    ds_valid = TensorDataset(ds_valid_, y_valid_)

    # Set optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_args["learning_rate"],
        weight_decay=train_args["weight_decay"],
    )

    # Set dataloader for train
    train_loader = DataLoader(
        ds_train,
        batch_size=train_args["batch_size"],
        shuffle=train_args["shuffle_train"],
    )

    # Set validation batch for evaluation
    valid_loader = DataLoader(ds_valid, batch_size=len(ds_valid_), shuffle=False)
    ds_valid_eval = next(iter(valid_loader))

    # Train model
    valid_loss_best = 9e15
    es_counter = 0
    model_best_ = copy.deepcopy(model)
    for _ in tqdm(
        range(1, train_args["max_epoch"] + 1),
        desc=f"Model No. xx",
        disable=train_args["disable_pbar"],
    ):
        _run_epoch(
            model,
            optimizer,
            train_loader,
            train_args["device"],
            train_args["loss"],
            train_args["output_dim"],
            train_args["criterion"],
        )
        valid_loss = _eval(
            model,
            ds_valid_eval,
            train_args["device"],
            train_args["loss"],
            train_args["output_dim"],
            train_args["valid_loss_metric"],
            train_args["valid_loss_flag"],
        )
        if valid_loss < valid_loss_best:
            valid_loss_best = valid_loss
            model_best_ = copy.deepcopy(model)
            es_counter = 0
        else:
            es_counter += 1
            if es_counter > train_args["early_stopping_patience"]:
                break
    model_best_.eval()

    return model_best_, valid_loss_best


def _run_epoch(model, optimizer, train_loader, device, loss, output_dim, criterion):
    """Run an epoch of the input model.

    Each epoch consists of steps that update the model and the optimizer.
    """
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.
        # Send to device
        data[0] = data[0].to(device)
        data[-1] = data[-1].to(device)
        # Feed-Foward
        out = model(data[0])  # Perform a single forward pass.
        target = data[-1].view(-1).to(torch.float32)  # Set target
        if loss == "categorical_crossentropy":
            target = target.to(torch.long)
        if output_dim == 1:
            out = out.view(-1).to(torch.float32)  # Reshape output
            target = target.to(torch.float32)  # Reshape target
        loss = criterion(out, target)  # Compute the loss.
        loss.backward()  # Scale the loss and backward pass
        optimizer.step()  # Update parameters


def _eval(
    model,
    ds_eval,
    device,
    loss,
    output_dim,
    valid_loss_metric,
    valid_loss_flag,
):
    """Run an evaluation of the input data on the input model.

    Returns the selected loss of the input data from the input model.
    """
    with torch.no_grad():
        model.eval()
        # Send to device
        ds_eval[0] = ds_eval[0].to(device)
        ds_eval[-1] = ds_eval[-1].to(device)
        # Feed-Foward
        out = model(ds_eval[0])
        target = ds_eval[-1].view(-1).to(torch.float32)
        if loss == "categorical_crossentropy":
            target = target.to(torch.long)
        if output_dim == 1:
            out = out.view(-1).to(torch.float32)
            target = target.to(torch.float32)
        valid_loss_metric.update(out, target)
        loss_eval = valid_loss_metric.compute()
        loss_eval = loss_eval.detach().item()
        if valid_loss_flag == "neg":
            loss_eval = -1 * loss_eval
        valid_loss_metric.reset()
    return loss_eval


def _generate_model_output(X, model_list, device):
    """Generate the output from the trained model."""

    # Set dataset and the test_loader
    ds_test = TensorDataset(X)
    test_loader = DataLoader(ds_test, batch_size=len(X), shuffle=False)

    # Obtain the batch to feed into the network
    ds_predict_eval = next(iter(test_loader))
    with torch.no_grad():
        # Send to device
        ds_predict_eval[0] = ds_predict_eval[0].to(device)
        ds_predict_eval[-1] = ds_predict_eval[-1].to(device)
        # Generate output
        out = [model(ds_predict_eval[0]).cpu().detach().numpy() for model in model_list]
    return np.average(out, weights=None, axis=0)


class MLPBase(BaseEstimator):
    """Base class for MLP."""

    def __init__(
        self,
        *,
        num_layers,
        hidden_dim,
        learning_rate,
        weight_decay,
        batch_size,
        dropout,
        val_size,
        shuffle_train,
        num_model,
        max_epoch,
        early_stopping_patience,
        n_jobs,
        device,
        random_state,
        disable_pbar,
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_size = val_size
        self.dropout = dropout
        self.shuffle_train = shuffle_train
        self.num_model = num_model
        self.max_epoch = max_epoch
        self.early_stopping_patience = early_stopping_patience
        self.n_jobs = n_jobs
        self.device = device
        self.random_state = random_state
        self.disable_pbar = disable_pbar

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

        # Preliminary settings
        self.is_fitted_ = False
        self.device_ = torch.device(self.device)
        self.X_ = X
        self.y_ = y
        self._set_task_specific_settings()

        X_, y_ = copy.deepcopy(X), copy.deepcopy(y)
        if isinstance(X, Tensor) == False:
            X_ = torch.tensor(X_, dtype=torch.float32)
        if isinstance(y, Tensor) == False:
            y_ = torch.tensor(y_, dtype=torch.float32)

        # Set the cv-splits
        splits = _set_train_valid_split(
            X_,
            y_,
            self.num_model,
            self.val_size,
            self.random_state,
            self.task_,
        )

        # Load model and optimizer
        model_train = self._load_model()
        model_train = model_train.to(self.device_)

        train_args = dict()
        train_args["learning_rate"] = self.learning_rate
        train_args["batch_size"] = self.batch_size
        train_args["early_stopping_patience"] = self.early_stopping_patience
        train_args["shuffle_train"] = self.shuffle_train
        train_args["max_epoch"] = self.max_epoch
        train_args["disable_pbar"] = self.disable_pbar
        train_args["device"] = self.device_
        train_args["weight_decay"] = self.weight_decay
        train_args["loss"] = self.loss
        train_args["output_dim"] = self.output_dim_
        train_args["criterion"] = self.criterion_
        train_args["valid_loss_metric"] = self.valid_loss_metric_
        train_args["valid_loss_flag"] = self.valid_loss_flag_

        # Fit model
        result_fit = Parallel(n_jobs=self.n_jobs)(
            delayed(_run_train_with_early_stopping)(
                X_, y_, split_index, model_train, train_args
            )
            for split_index in splits
        )

        # Store the required results that may be used later
        self.model_list_ = [model for (model, _) in result_fit]
        self.valid_loss_ = [valid_loss for (_, valid_loss) in result_fit]
        self.is_fitted_ = True

        return self

    def _generate_output(self, X, model_list, device):
        """Generate the output from the trained model.

        Returns the output (prediction) of input X.
        """

        out = _generate_model_output(X, model_list, device)

        # Change if the task is classification
        if self.loss == "binary_crossentropy":
            out = 1 / (1 + np.exp(-out))
        elif self.loss == "categorical_crossentropy":
            out = softmax(out, axis=1)

        # Control for nulls in prediction
        if np.isnan(out).sum() > 0:
            mean_pred = np.mean(self.y_)
            out[np.isnan(out)] = mean_pred

        if out.ndim == 2 and out.shape[1] == 1:
            out = out.squeeze(axis=1)  # we don't want to squeeze first axis

        return out

    def _set_task_specific_settings(self):
        """Set task specific settings for regression and classfication."""

        if self._estimator_type == "regressor":
            self.task_ = "regression"
            if self.loss == "squared_error":
                self.criterion_ = torch.nn.MSELoss()
            elif self.loss == "absolute_error":
                self.criterion_ = torch.nn.L1Loss()
            if self.scoring == "squared_error":
                self.valid_loss_metric_ = MeanSquaredError()
                self.valid_loss_flag_ = "pos"
            elif self.scoring == "r2_score":
                self.valid_loss_metric_ = R2Score()
                self.valid_loss_flag_ = "neg"
            self.output_dim_ = 1
        elif self._estimator_type == "classifier":
            self.task_ = "classification"
            self.classes_ = np.unique(self.y_)
            if self.loss == "binary_crossentropy":
                self.criterion_ = torch.nn.BCEWithLogitsLoss()
                self.output_dim_ = 1
                if self.scoring == "auroc":
                    self.valid_loss_metric_ = BinaryAUROC()
                    self.valid_loss_flag_ = "neg"
                elif self.scoring == "binary_entropy":
                    self.valid_loss_metric_ = BinaryNormalizedEntropy(from_logits=True)
                    self.valid_loss_flag_ = "neg"
                elif self.scoring == "auprc":
                    self.valid_loss_metric_ = BinaryAUPRC()
                    self.valid_loss_flag_ = "neg"
            elif self.loss == "categorical_crossentropy":
                self.criterion_ = torch.nn.CrossEntropyLoss()
                self.output_dim_ = len(np.unique(self.y_))
                self.valid_loss_metric_ = MulticlassAUROC(num_classes=self.output_dim_)
                self.valid_loss_flag_ = "neg"
        self.valid_loss_metric_.to(self.device_)

    def _load_model(self):
        """Load the MLP model for training.

        Returns the model that can be used for training.
        """

        # Set seed for torch - for reproducibility
        random_state = check_random_state(self.random_state)
        model_seed = random_state.randint(10000)
        torch.manual_seed(model_seed)

        model_config = dict()
        model_config["input_dim"] = self.X_.shape[1]
        model_config["hidden_dim"] = self.hidden_dim
        model_config["output_dim"] = self.output_dim_
        model_config["dropout"] = self.dropout
        model_config["num_layers"] = self.num_layers
        return MLP_Model(**model_config)


class TARTE_MLPRegressor(RegressorMixin, MLPBase):
    """TARTE MLP Regressor for Regression tasks.

    Parameters
    ----------
    loss : {'squared_error', 'absolute_error'}, default='squared_error'
        The loss function used for backpropagation.
    scoring : {'r2_score', 'squared_error'}, default='r2_score'
        The scoring function used for validation.
    num_layers : int, default=1
        The number of layers for the MLP model.
    hidden_dim : int, default=256
        The dimension of the hidden layers of the model.
    learning_rate : float, default=1e-3
        The learning rate of the model. The model uses AdamW as the optimizer
    weight_decay : float, default=0.2
        The weight decay of the AdamW optimizer.
    dropout : float, default=0.2
        The dropout rate for training
    batch_size : int, default=128
        The batch size used for training
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    shuffle_train : bool, default=False
        Indicates whether to shuffle the train data for batch.
    num_model : int, default=1
        The total number of models used for Bagging strategy
    max_epoch : int or None, default=500
        The maximum number of epoch for training
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    device : {"cpu", "cuda"}, default="cuda",
        The device used for the estimator.
    random_state : int or None, default=0
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    disable_pbar : bool, default=True
        Indicates whether to show progress bars for the training process.
    """

    def __init__(
        self,
        *,
        loss: str = "squared_error",
        scoring: str = "r2_score",
        num_layers: int = 2,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        dropout: float = 0.2,
        batch_size: int = 128,
        val_size: float = 0.1,
        shuffle_train: bool = False,
        num_model: int = 1,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 40,
        n_jobs: int = 1,
        device: str = "cpu",
        random_state: int = 0,
        disable_pbar: bool = True,
    ):
        super(TARTE_MLPRegressor, self).__init__(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            shuffle_train=shuffle_train,
            batch_size=batch_size,
            val_size=val_size,
            num_model=num_model,
            max_epoch=max_epoch,
            early_stopping_patience=early_stopping_patience,
            n_jobs=n_jobs,
            device=device,
            random_state=random_state,
            disable_pbar=disable_pbar,
        )

        self.loss = loss
        self.scoring = scoring

    def predict(self, X):

        check_is_fitted(self, "is_fitted_")

        X_ = copy.deepcopy(X)

        if isinstance(X_, Tensor) == False:
            X_ = torch.tensor(X, dtype=torch.float32)

        return self._generate_output(X_, self.model_list_, self.device_)


class TARTE_MLPClassifier(ClassifierMixin, MLPBase):
    """TARTE MLP Classifier for Classification tasks.

    Parameters
    ----------
    loss : {'binary_crossentropy', 'categorical_crossentropy'}, default='binary_crossentropy'
        The loss function used for backpropagation.
    scoring : {'auroc', 'auprc', 'binary_entropy'}, default='auroc'
        The scoring function used for validation.
    num_layers : int, default=1
        The number of layers for the MLP model.
    hidden_dim : int, default=256
        The dimension of the hidden layers of the model.
    learning_rate : float, default=1e-3
        The learning rate of the model. The model uses AdamW as the optimizer
    weight_decay : float, default=0.2
        The weight decay of the AdamW optimizer.
    dropout : float, default=0.2
        The dropout rate for training
    batch_size : int, default=128
        The batch size used for training
    val_size : float, default=0.1
        The size of the validation set used for early stopping
    shuffle_train : bool, default=False
        Indicates whether to shuffle the train data for batch.
    num_model : int, default=1
        The total number of models used for Bagging strategy
    max_epoch : int or None, default=500
        The maximum number of epoch for training
    early_stopping_patience : int or None, default=40
        The early stopping patience when early stopping is used.
        If set to None, no early stopping is employed
    n_jobs : int, default=1
        Number of jobs to run in parallel. Training the estimator the score are parallelized
        over the number of models.
    device : {"cpu", "cuda"}, default="cuda",
        The device used for the estimator.
    random_state : int or None, default=0
        Pseudo-random number generator to control the train/validation data split
        if early stoppingis enabled, the weight initialization, and the dropout.
        Pass an int for reproducible output across multiple function calls.
    disable_pbar : bool, default=True
        Indicates whether to show progress bars for the training process.
    """

    def __init__(
        self,
        *,
        loss: str = "binary_crossentropy",
        scoring: str = "auroc",
        num_layers: int = 2,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        dropout: float = 0.2,
        batch_size: int = 128,
        val_size: float = 0.1,
        shuffle_train: bool = False,
        num_model: int = 1,
        max_epoch: int = 200,
        early_stopping_patience: Union[None, int] = 40,
        n_jobs: int = 1,
        device: str = "cpu",
        random_state: int = 0,
        disable_pbar: bool = True,
    ):
        super(TARTE_MLPClassifier, self).__init__(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            shuffle_train=shuffle_train,
            batch_size=batch_size,
            val_size=val_size,
            num_model=num_model,
            max_epoch=max_epoch,
            early_stopping_patience=early_stopping_patience,
            n_jobs=n_jobs,
            device=device,
            random_state=random_state,
            disable_pbar=disable_pbar,
        )

        self.loss = loss
        self.scoring = scoring

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")

        if self.loss == "binary_crossentropy":
            return np.round(self.predict_proba(X))
        elif self.loss == "categorical_crossentropy":
            return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        return self._get_predict_prob(X)

    def decision_function(self, X):
        decision = self.predict_proba(X)
        return decision

    def _get_predict_prob(self, X):
        # Obtain the predicitve output
        X_ = copy.deepcopy(X)
        if isinstance(X_, Tensor) == False:
            X_ = torch.tensor(X, dtype=torch.float32)
        return self._generate_output(X_, self.model_list_, self.device_)
