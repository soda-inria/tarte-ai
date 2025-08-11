"""Preprocessing a table suitable for TARTE."""

import torch
import numpy as np
import pandas as pd
import json
import gc
from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.pipeline import Pipeline
from skrub import to_datetime
from tarte_ai.tarte_model import TARTE_Pretrain_NN
from tarte_ai.tarte_utils import load_tarte_pretrain_model, load_fasttext


class TARTE_TablePreprocessor(TransformerMixin, BaseEstimator):
    """Preprocessor from tables to inputs for transformers.

    Parameters
    ----------
    fasttext_model_path : str or None, default=None
        Path to the FastText model file.
    """

    def __init__(
        self,
        *,
        fasttext_model_path: str = None,
    ):
        super(TARTE_TablePreprocessor, self).__init__()

        self.fasttext_model_path = fasttext_model_path

    def fit(self, X, y=None):
        """
        Fit function used for the TARTE_TablePreprocessor.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data to fit.
        y : array-like, optional
            Target values, by default None.

        Returns
        -------
        self : TARTE_TablePreprocessor
            Fitted transformer.
        """

        if isinstance(X, pd.DataFrame) == False:
            X = pd.DataFrame(X)
            col_names = [f"Column_{i}" for i in range(X.shape[1])]
            X = X.set_axis(col_names, axis="columns")

        X_ = X.replace("\n", " ", regex=True).copy()
        self.is_fitted_ = False
        self.y_ = y
        if not hasattr(self, "lm_model_"):
            self._load_lm_model()

        # Load language_model
        if hasattr(self, "lm_model_") == False:
            self._load_lm_model()

        # Preprocess for Datetime information
        dat_col_names = []
        for col in X_:
            if pd.api.types.is_datetime64_any_dtype(to_datetime(X_[col])):
                datetime = pd.to_datetime(X_[col].astype("datetime64[s]"))
                X_[col] = datetime.dt.strftime("%Y").astype(float)
                dat_col_names.append(col)
        self.dat_col_names_ = dat_col_names

        # Use original column names without lowercasing to avoid mismatches
        cat_col_names = X_.select_dtypes(include="object").columns.str.replace(
            "\n", " ", regex=True
        )
        self.cat_col_names_ = list(set(cat_col_names) - set(dat_col_names))
        num_col_names = X_.select_dtypes(exclude="object").columns.str.replace(
            "\n", " ", regex=True
        )
        self.num_col_names_ = list(set(num_col_names) - set(dat_col_names))
        self.col_names_ = (
            self.cat_col_names_ + self.num_col_names_ + self.dat_col_names_
        )

        # Set max-pad-size
        self.max_pad_size_ = len(self.col_names_)

        # Set transformers for numerical and datetime
        self.num_transformer_ = PowerTransformer().set_output(transform="pandas")
        self.dat_transformer_ = Pipeline(
            [
                ("scale", RobustScaler()),
                ("power", PowerTransformer()),
            ]
        ).set_output(transform="pandas")

        # Ensure numerical columns exist before fitting the transformer
        num_cols_exist = [col for col in self.num_col_names_ if col in X_.columns]
        if num_cols_exist:
            self.num_transformer_.fit(X_[num_cols_exist])

        dat_cols_exist = [col for col in self.dat_col_names_ if col in X_.columns]
        if dat_cols_exist:
            self.dat_transformer_.fit(X_[dat_cols_exist])

        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        """
        Apply Table2GraphTransformer to each row of the data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data to transform.
        y : array-like, optional
            Target values, by default None.

        Returns
        -------
        data_preprocessed : list
            List of preprocessed input (containing x, edge_attr, mask).
        """

        if isinstance(X, pd.DataFrame) == False:
            X = pd.DataFrame(X)
            col_names = [f"Column_{i}" for i in range(X.shape[1])]
            X = X.set_axis(col_names, axis="columns")

        X_ = X.replace("\n", " ", regex=True).copy()
        num_data = X_.shape[0]
        y_ = (
            torch.tensor(self.y_, dtype=torch.float32).reshape((num_data, 1))
            if self.y_ is not None
            else None
        )

        # Preprocess for year information
        for col in self.dat_col_names_:
            datetime = pd.to_datetime(X_[col].astype("datetime64[s]"))
            X_[col] = datetime.dt.strftime("%Y").astype(float)

        X_categorical = X_[self.cat_col_names_].copy()
        X_numerical = X_[self.num_col_names_].copy()
        X_datetime = X_[self.dat_col_names_].copy()

        cat_names = (
            pd.melt(X_categorical)["value"].dropna().astype(str).str.lower().unique()
        )
        names_total = np.unique(np.hstack([self.col_names_, cat_names]))
        name_dict = {name: idx for idx, name in enumerate(names_total)}

        name_attr_total = self._transform_names(names_total)

        # Use the original numerical column names for transformation
        num_cols_exist = [col for col in self.num_col_names_ if col in X_.columns]
        if num_cols_exist:
            X_numerical = self.num_transformer_.transform(X_numerical)

        dat_cols_exist = [col for col in self.dat_col_names_ if col in X_.columns]
        if dat_cols_exist:
            X_datetime = self.dat_transformer_.transform(X_datetime)

        data_preprocessed = [
            self._preprocess_data(
                X_categorical.iloc[idx],
                X_numerical.iloc[idx],
                X_datetime.iloc[idx],
                name_attr_total,
                name_dict,
                y_,
                idx,
            )
            for idx in range(num_data)
        ]

        if self.y_ is not None:
            self.y_ = None

        gc.collect()

        return data_preprocessed

    def _load_lm_model(self):
        """
        Load the language model for features of nodes and edges.
        """

        # Without designated path
        if self.fasttext_model_path is None:
            self.lm_model_ = load_fasttext()
        # With designated path
        else:

            import fasttext

            self.lm_model_ = fasttext.load_model(self.fasttext_model_path)
        
        self.n_components_ = 300

    def _transform_names(self, names_total):
        """
        Obtain the feature for a given list of string values.

        Parameters
        ----------
        names_total : array-like
            List of string values.

        Returns
        -------
        name_features : np.ndarray
            Transformed features for names.
        """

        return np.array(
            [self.lm_model_.get_sentence_vector(name) for name in names_total],
            dtype=np.float32,
        )

    def _preprocess_data(
        self, data_cat, data_num, data_dat, name_attr_total, name_dict, y, idx
    ):
        """Process to suitable PyTorch tensors.

        Parameters
        ----------
        data_cat : pandas.Series
            Categorical data for a single instance.
        data_num : pandas.Series
            Numerical data for a single instance.
        data_dat : pandas.Series
            Datetime data for a single instance.
        name_attr_total : np.ndarray
            Transformed features for names.
        name_dict : dict
            Dictionary mapping names to indices.
        y : torch.Tensor or None
            Target values.
        idx : int
            Index of the instance.

        Returns
        -------

        """
        data_cat = data_cat.dropna()
        if data_cat.shape[0] > 0:
            data_cat = data_cat.str.lower()
        data_num = data_num.dropna()
        data_dat = data_dat.dropna()

        edge_attr_cat = np.array(
            [name_attr_total[name_dict[col]] for col in data_cat.index],
            dtype=np.float32,
        )
        edge_attr_num = np.array(
            [name_attr_total[name_dict[col]] for col in data_num.index],
            dtype=np.float32,
        )
        edge_attr_dat = np.array(
            [name_attr_total[name_dict[col]] for col in data_dat.index],
            dtype=np.float32,
        )

        x_cat = torch.tensor(
            np.array([name_attr_total[name_dict[val]] for val in data_cat]),
            dtype=torch.float32,
        )
        x_num = torch.tensor(
            data_num.values[:, None] * edge_attr_num, dtype=torch.float32
        )
        x_dat = torch.tensor(
            data_dat.values[:, None] * edge_attr_dat, dtype=torch.float32
        )

        if x_cat.size(0) == 0:
            x_cat = torch.empty((0, self.n_components_), dtype=torch.float32)
            edge_attr_cat = torch.empty((0, self.n_components_), dtype=torch.float32)
        if x_num.size(0) == 0:
            x_num = torch.empty((0, self.n_components_), dtype=torch.float32)
            edge_attr_num = torch.empty((0, self.n_components_), dtype=torch.float32)
        if x_dat.size(0) == 0:
            x_dat = torch.empty((0, self.n_components_), dtype=torch.float32)
            edge_attr_dat = torch.empty((0, self.n_components_), dtype=torch.float32)

        # combined node/edge attributes
        x = torch.vstack((x_cat, x_num, x_dat))
        x = torch.vstack((torch.ones((1, x.size(1))), x))
        edge_attr = torch.tensor(
            np.vstack((edge_attr_cat, edge_attr_num, edge_attr_dat)),
            dtype=torch.float32,
        )
        edge_attr = torch.vstack((torch.ones((1, self.n_components_)), edge_attr))

        # Process for the paddings
        pad_size = self.max_pad_size_ - x.size(0) + 1
        x_mask = torch.zeros((self.max_pad_size_ + 1,), dtype=bool)

        if pad_size > 0:
            x_mask[-pad_size:] = True

        pad_emb = -1 * torch.ones((pad_size, self.n_components_))
        x = torch.vstack((x, pad_emb))
        edge_attr = torch.vstack((edge_attr, pad_emb))

        y_ = y[idx].clone() if y is not None else torch.tensor([])

        # row index
        r_idx = idx

        return r_idx, x, edge_attr, x_mask, y_


class TARTE_TableEncoder(TransformerMixin, BaseEstimator):
    """TARTE Table Encoder from TARTE compatible inputs to embeddings.

    Parameters
    ----------
    fasttext_model_path : str or None, default=None
        Path to the FastText model file.

        
    """

    def __init__(
        self,
        *,
        pretrained_model_path: Union[None, str] = None,
        pretrained_model_config_path: Union[None, str] = None,
        dim_embedding: int = 768,
        layer_index: Union[int, list, str] = "all",
        device: str = "cpu",
    ):

        self.pretrained_model_path = pretrained_model_path
        self.pretrained_model_config_path = pretrained_model_config_path
        self.dim_embedding = dim_embedding
        self.layer_index = layer_index
        self.device = device

    def fit(self, X, y=None):

        # Set preliminaries
        self.is_fitted_ = False
        self.numerical_preprocessor_ = SimpleImputer(strategy="mean")
        self.device_ = torch.device(self.device)

        # Load the pretrain weights
        # With designated path
        if self.pretrained_model_path is not None:
            self.pretrain_model_dict_ = torch.load(
                self.pretrained_model_path,
                map_location='cpu',
                weights_only=True,
                mmap=True,
            )
            filename = open(self.pretrained_model_config_path)
            self.pretrained_model_configs_ = json.load(filename)
            filename.close()
        # Without designated path
        else:
            self.pretrain_model_dict_, self.pretrained_model_configs_ = load_tarte_pretrain_model()

        # Set the layer index accordingly
        if isinstance(self.layer_index, int):
            self.layer_index_ = [self.layer_index]
        elif self.layer_index == 'all':
            self.layer_index_ = [
                x
                for x in range(self.pretrained_model_configs_["num_layers_transformer"])
            ]
        else:
            self.layer_index_ = self.layer_index

        return self

    def transform(self, X, y=None):
        return self._extract_tarte_embeddings(X)

    def _extract_tarte_embeddings(self, X):

        # Model configuration
        model_config = dict()
        model_config["dim_input"] = 300
        model_config["num_heads"] = 24
        model_config["num_layers_transformer"] = 3
        model_config["dropout"] = 0.1
        model_config["dim_transformer"] = 768
        model_config["dim_feedforward"] = 2048

        for key in model_config.keys():
            model_config[key] = self.pretrained_model_configs_[key]

        # Preprocess the data
        x_ = torch.stack([x for (_, x, _, _, _) in X])
        edge_attr_ = torch.stack([edge_attr for (_, _, edge_attr, _, _) in X])
        mask_ = torch.stack([mask for (_, _, _, mask, _) in X])

        # Move to device
        x_ = x_.to(self.device_)
        edge_attr_ = edge_attr_.to(self.device_)
        mask_ = mask_.to(self.device_)

        X_total = [
            self._extract_per_layer(
                x_,
                edge_attr_,
                mask_,
                model_config,
                layer_idx + 1,
            )
            for layer_idx in self.layer_index_
        ]
        X_total = np.array(X_total)
        X_total = X_total.mean(axis=0)

        if self.is_fitted_:
            X_total = self.numerical_preprocessor_.transform(X_total)
        else:
            X_total = self.numerical_preprocessor_.fit_transform(X_total, y=None)

        self.is_fitted_ = True

        return np.array(X_total, dtype=np.float32)

    def _extract_per_layer(
        self,
        x,
        edge_attr,
        mask,
        model_config,
        num_layers_transformer,
    ):

        model_config["num_layers_transformer"] = num_layers_transformer
        model_config["dim_projection"] = self.dim_embedding
        model = TARTE_Pretrain_NN(**model_config)

        max_dim_ = np.max(self.pretrained_model_configs_["dim_projection"])
        if self.dim_embedding == max_dim_:
            model.tarte_linear = torch.nn.ModuleDict({"tab_enc": torch.nn.Identity()})
        else:
            remove_idx = [2]
            remove_idx += [x for x in range(4, 8)]
            for idx in remove_idx:
                model.tarte_linear[f"proj_{self.dim_embedding}"][
                    idx
                ] = torch.nn.Identity()

        model = model.to(self.device_)
        model.load_state_dict(self.pretrain_model_dict_, strict=False)
        model.layer_norm = torch.nn.LayerNorm(model_config["dim_transformer"])

        with torch.no_grad():
            model.eval()
            X_ = model(x, edge_attr, mask)[0]
            X_ = X_.detach().to("cpu").numpy()

        return X_
