"""
Knowledge graph preprocessor for CARTE pretraining.
"""

import torch
import numpy as np
from torch.utils.data import Dataset


def _remove_duplicates(edge_index):
    """Function to remove duplicates in a graph defined by the edge_index."""

    nnz = edge_index.size(1)
    num_nodes = edge_index.max().item() + 1
    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[0]
    idx[1:].mul_(num_nodes).add_(edge_index[1])
    idx[1:], perm = torch.sort(
        idx[1:],
    )
    mask = idx[1:] > idx[:-1]
    edge_index = edge_index[:, perm]
    edge_index = edge_index[:, mask]
    return edge_index


def _reduce_by_max_rels(edge_index, edge_type, max_num_per_rel):
    """Randomly reduces the edge_index and edge_type with max. number per relation."""

    edge_type_sorted, idx_sorted = edge_type.sort()
    edge_index_sorted = edge_index[:, idx_sorted]

    # Obtain the mask
    edge_count = np.bincount(edge_type_sorted)
    edge_count = edge_count[edge_count.nonzero()]
    if edge_count.max() > max_num_per_rel:
        ptr = np.zeros(edge_count.shape[0] + 1)
        ptr[1:] = np.cumsum(edge_count)
        edge_count[edge_count > max_num_per_rel] = max_num_per_rel
        mask = [
            np.random.choice(
                np.arange(ptr[i], ptr[i + 1]), size=edge_count[i], replace=False
            )
            for i in range(edge_count.shape[0])
        ]
        mask = np.hstack(mask)
        mask = torch.from_numpy(mask).to(torch.long)
        return edge_index_sorted[:, mask], edge_type_sorted[mask]
    else:
        return edge_index, edge_type


def _sample_by_min_rels(edge_index, edge_type, edge_num_mask, p_num, min_rel):
    """Randomly selects edge_type to keep defined by the number of minium relations."""

    edge_type_list = edge_type.unique()
    n_num = edge_num_mask[edge_type_list].nonzero().size(0)
    weights = torch.ones(
        edge_type_list.size(0),
    )
    if n_num != 0:
        weights[edge_num_mask[edge_type_list]] = p_num / n_num
        weights[~edge_num_mask[edge_type_list]] = (1 - p_num) / (
            edge_type_list.size(0) - n_num
        )
    idx_keep = torch.multinomial(weights, min_rel, replacement=False)
    mask_keep = torch.isin(edge_type, edge_type_list[idx_keep])
    return edge_index[:, mask_keep], edge_type[mask_keep]


class TARTEKGPreprocessor:

    def __init__(
        self,
        data_kg_dir: str,
        num_hops: int = 1,
        num_entity: int = 2,
        num_pos: int = 1,
        min_rels: int = 3,
        max_num_per_rel: int = 1,
    ):
        """TARTE knowledge graph preprocessor.

        Preprocessor extract subgraphs from knowledge graph and set appropriate  inputs for transformer.

        Parameters
        ----------
        data_kg_dir : str
            The directory of the store knowledge graph.
        num_hops : int, default=1,
            The number of hops to extract the subgraph of an entity.
        num_entity : int, default=2,
            The number of entity indices to extract for each iteration.
        num_pos : int, default=1,
            The number of positives to generate.
        min_rels : int, default=3,
            The minimum number of unique relations for each entity for preprocessing.
        max_num_per_rel : int, default=1,
            The maximum number of relations for each unique relation.
        """

        super(TARTEKGPreprocessor, self).__init__()

        self.data_kg_dir = data_kg_dir
        self.num_hops = num_hops
        self.num_entity = num_entity
        self.num_pos = num_pos
        self.min_rels = min_rels
        self.max_num_per_rel = max_num_per_rel

        self._set_preprocessor()

    def _set_preprocessor(self):
        """Load knowledge graph and set preprocessor."""

        # Load knowledge graph data
        self.edge_attr_total = torch.load(
            f"{self.data_kg_dir}/edge_attr_total.pt",
            weights_only=True,
            mmap=True,
        )
        self.edge_num_mask = torch.load(
            f"{self.data_kg_dir}/edge_num_mask.pt",
            weights_only=True,
            mmap=True,
        )
        self.x_total = torch.load(
            f"{self.data_kg_dir}/x_total.pt",
            weights_only=True,
            mmap=True,
        )
        self.edge_index = torch.load(
            f"{self.data_kg_dir}/edge_index.pt",
            weights_only=True,
            mmap=True,
        )
        self.edge_type = torch.load(
            f"{self.data_kg_dir}/edge_type.pt",
            weights_only=True,
            mmap=True,
        )

        # Set preliminaries
        self.edge_type_list = self.edge_type.unique().numpy()  # edge_types
        self.num_total_entity = self.edge_index.unique().size(0)
        self.dim = self.x_total.size(1)  # Dimension of data

        # Number of unique relations
        count_index = torch.vstack([self.edge_index[0], self.edge_type])
        count_index = _remove_duplicates(count_index)
        self.count_rels = count_index[0].bincount()

        # Required slices
        # slice_by_head used for extracting 1-hop subgraph of selected entity
        slice_by_head = torch.cumsum(torch.bincount(self.edge_index[0]), dim=0)
        slice_by_head_ = torch.hstack(
            [torch.zeros(1, dtype=torch.long), slice_by_head[:-1]]
        )
        slice_by_head = torch.vstack((slice_by_head_, slice_by_head))
        self.slice_by_head = slice_by_head.transpose(-1, 0)

        # slice_by_rel used sampling entities with at least n number of relations
        sorted, self.idx_sorted_rel = self.count_rels.sort()
        sorted_count = sorted.bincount()
        self.slice_by_rel = torch.cumsum(sorted_count, dim=0)
        num_cut = (
            sorted_count.reshape((1, sorted_count.size(0)))
            .fliplr()
            .cumsum(dim=1)
            .view(-1)
        )
        num_cut = max(1, (num_cut > self.num_entity).nonzero().view(-1)[0])

        self.max_rels_ = self.slice_by_rel.size(0) - num_cut + 1

        return None

    def __call__(self):
        """Call used for extract batch."""

        # Extract center indices
        center_indices = self._sample_index()
        self.center_indices_ = center_indices

        # Change the type of center indices if condition not satisfied.
        if isinstance(center_indices, int):
            center_indices = [center_indices]
        if isinstance(center_indices, list) == False:
            center_indices = center_indices.tolist()

        return [self._preprocess_entity(ent_idx) for ent_idx in center_indices]

    def _sample_index(self):
        """Samples the indices and controls for the probability after sampling."""

        num_rel = torch.randint(self.min_rels, self.max_rels_, (1,))
        idx_candidate = self.idx_sorted_rel[self.slice_by_rel[num_rel - 1] :]
        idx_sample = idx_candidate[torch.randperm(idx_candidate.size(0))][
            : self.num_entity
        ]
        self.num_rel = num_rel.item()
        self.max_pad_size = self.count_rels[idx_sample].max() * self.max_num_per_rel

        return idx_sample.tolist()

    def _preprocess_entity(self, ent_idx):
        """Creates a sample (positives and negatives) for an entitiy of interest."""

        # Obatin edge_index, and edge_type for the ent_idx
        edge_index = self.edge_index[
            :, self.slice_by_head[ent_idx, 0] : self.slice_by_head[ent_idx, 1]
        ].clone()
        edge_type = self.edge_type[
            self.slice_by_head[ent_idx, 0] : self.slice_by_head[ent_idx, 1]
        ].clone()

        # Extract list of entities in the subgraph for later usage
        self.ent_list_ = edge_index.unique()

        # Control for number of rels
        edge_index, edge_type = _reduce_by_max_rels(
            edge_index,
            edge_type,
            self.max_num_per_rel,
        )
        p_num = 0.5
        edge_index, edge_type = _sample_by_min_rels(
            edge_index,
            edge_type,
            self.edge_num_mask,
            p_num,
            self.num_rel,
        )

        # Generate original and positives
        data_original = [
            self._preprocess_transformer(
                ent_idx,
                edge_index,
                edge_type,
                "original",
            )
        ]
        data_pos = [
            self._preprocess_transformer(
                ent_idx,
                edge_index,
                edge_type,
                "pos",
            )
            for _ in range(self.num_pos)
        ]

        return self._collate(data_original + data_pos)

    def _preprocess_transformer(
        self,
        ent_idx,
        edge_index,
        edge_type,
        perturb_type="original",
    ):
        """Function to process an extracted subgraph appropriate for transformers.
        It sets the target depending on the positive or negative.
        """

        edge_index_, edge_type_ = edge_index.clone(), edge_type.clone()

        if perturb_type == "original":
            y = -1 * torch.ones((1,))
        elif perturb_type == "pos":
            p = 0.2 if edge_type_.size(0) >= 13 else 0.1
            num_col_perturb = 2 if torch.bernoulli(torch.tensor(p)) == 1 else 1
            perturb_idx = torch.from_numpy(
                np.random.choice(np.arange(edge_type_.size(0)), num_col_perturb)
            )
            edge_index_, edge_type_ = self._create_neg(
                edge_index,
                edge_type,
                perturb_idx,
            )
            y = torch.ones((1,))

        # Required masks
        mask_x = edge_index_[1]
        mask_rel = edge_type_

        # Node features
        x = torch.ones(mask_x.size(0) + 1, self.x_total.size(1))
        x[1:, :] = self.x_total[mask_x]

        # Edge features
        edge_attr = torch.ones(mask_rel.size(0) + 1, self.x_total.size(1))
        edge_attr[1:, :] = self.edge_attr_total[mask_rel]

        # Masks for transformers
        pad_size = self.max_pad_size - x.size(0) + 1
        pad_mask = torch.zeros((self.max_pad_size + 1,), dtype=bool)

        if pad_size > 0:
            pad_mask[-pad_size:] = True

        pad_emb = -1 * torch.ones((pad_size, x.size(1)))
        x = torch.vstack((x, pad_emb))
        edge_attr = torch.vstack((edge_attr, pad_emb))

        ent_idx_ = torch.tensor([ent_idx])

        return x, edge_attr, pad_mask, y, ent_idx_

    def _create_neg(self, edge_index, edge_type, perturb_idx):
        """Create a negative leaflets by replacing an entity(node) with an entity in same relation."""

        replace_ent = torch.zeros((0,), dtype=torch.long)
        for _ in perturb_idx:
            replace_ent_ = torch.randint(self.num_total_entity, (1,))
            while torch.isin(self.ent_list_, replace_ent_).nonzero().size(0) != 0:
                replace_ent_ = torch.randint(self.num_total_entity, (1,))
            replace_ent = torch.hstack((replace_ent, replace_ent_))
        edge_index_ = edge_index.clone()
        edge_index_[1, perturb_idx] = replace_ent

        return edge_index_, edge_type

    def _collate(self, sample):
        """Collating function to set the extract list of samples in an appropriate batch."""

        x = torch.stack([x for (x, _, _, _, _) in sample], dim=0)
        edge_attr = torch.stack(
            [edge_attr for (_, edge_attr, _, _, _) in sample], dim=0
        )
        mask = torch.stack([mask for (_, _, mask, _, _) in sample], dim=0)
        y = torch.stack([y for (_, _, _, y, _) in sample], dim=0)
        ent_idx = torch.stack([ent_idx for (_, _, _, _, ent_idx) in sample])

        return x, edge_attr, mask, y, ent_idx


class KGDataset(Dataset):
    """PyTorch Dataset used for dataloader."""

    def __init__(self, num_steps, kg_preprocessor):

        self.num_steps = num_steps
        self.kg_preprocessor = kg_preprocessor

    def __len__(self):
        """The length of the dataloader (an epoch) for the pretraining.
        For pretraining, it does not have the concepts of epoch, only defined by the steps.
        """
        return self.num_steps

    def __getitems__(self, idx):
        """Extract the preprocessed sample. The idx parameter is ignored, but only required for __getitems__."""
        return self.kg_preprocessor()


def kg_batch_collate(batch):
    """Collate function used for the dataloader to set appropriate dimensions for the transformer."""

    x = torch.cat([item[0] for item in batch])
    edge_attr = torch.cat([item[1] for item in batch])
    mask = torch.cat([item[2] for item in batch])

    y_global = torch.cat([item[3].T for item in batch])
    if (y_global == 0).sum() == 0:
        y_global = torch.block_diag(
            *[torch.ones((y_global.size(1), y_global.size(1))) for _ in batch]
        )
        y_global = y_global.to(torch.float)
    else:
        y_global = y_global[y_global != -1]

    original_mask = torch.cat([item[3] for item in batch])
    original_mask[original_mask > -1] = 0
    original_mask[original_mask < 0] = 1
    original_mask = original_mask.to(torch.bool).view(-1)

    return x, edge_attr, mask, original_mask, y_global
