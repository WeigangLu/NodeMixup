import torch
from torch_geometric.utils import degree, dense_to_sparse, to_dense_adj
import numpy as np
from torch.utils.data import WeightedRandomSampler
from utils import sharpen


def _normalize(vec):
    return vec / vec.sum()


def _check_results(value):
    return torch.isnan(value) or torch.isinf(value)


def cosine_similarity_matrix(A, B):
    A_norm = torch.norm(A, dim=1, keepdim=True)
    B_norm = torch.norm(B, dim=1, keepdim=True)
    # normalize
    A_normalized = A / A_norm
    B_normalized = B / B_norm
    # cosine similarity
    similarity_matrix = torch.mm(A_normalized, B_normalized.t())
    return similarity_matrix


class Mixer(object):

    def __init__(self, t_idx, un_idx, nclass, beta_d=0.5, beta_s=0.5, temp=0.1, train_size=20, alpha=1., gamma=0.7,
                 device=torch.device('cpu')):
        self.t_idx = t_idx
        self.un_idx = un_idx
        self.alpha = alpha
        self.gamma = gamma
        self.nclass = nclass
        self.beta_d = beta_d
        self.beta_s = beta_s
        self.temp = temp

        self.E = None
        self.deg_prob = None
        self.select_l_idx = None
        self.nb_dist = None

        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')

        self.device = device

    def _generate_lam(self):
        if self.alpha > 0.:
            return np.random.beta(self.alpha, self.alpha)
        else:
            return 1.

    def mixup_loss(self, eq_mixup_logits, neq_mixup_logits, mixup_dict):

        eq_mixup_loss = self.criterion(eq_mixup_logits[mixup_dict['eq_idx']].softmax(-1),
                                       mixup_dict['eq_mixup_y'][mixup_dict['eq_idx']])
        neq_mixup_loss = self.criterion(neq_mixup_logits.softmax(-1),
                                        mixup_dict['neq_mixup_y'])

        eq_mixup_loss = 0. if _check_results(eq_mixup_loss) else eq_mixup_loss
        neq_mixup_loss = 0. if _check_results(neq_mixup_loss) else neq_mixup_loss

        return eq_mixup_loss, neq_mixup_loss

    def _training_set_prepare(self, target, adj):
        if self.select_l_idx is not None and self.deg_prob is not None:
            return

        labeled_set = dict()
        for i in range(self.nclass):
            labeled_set[i] = self.t_idx[target[self.t_idx] == i]

        self.select_l_idx = torch.cat(list(labeled_set.values()), dim=0)

    def vanilla_mixup(self, x, target, train_idx):
        lam = self._generate_lam()
        perm_index = train_idx[torch.randperm(train_idx.shape[0])]
        x[train_idx] = lam * x[train_idx] + (1 - lam) * x[perm_index]
        target[train_idx] = lam * target[train_idx] + (1 - lam) * target[perm_index]

        return x, target

    def neighbour_distribution(self, y, adj):
        import torch_scatter
        row, col = adj
        nb_dist = torch_scatter.scatter_mean(y[col], row, dim=0, dim_size=y.shape[0]).detach()
        self.nb_dist = sharpen(nb_dist, temp=self.temp)

    def mixup_data(self, x, y, adj):
        if self.E is None:
            self.E = torch.empty(size=(2, 0), dtype=torch.int64).to(self.device)
        self.neighbour_distribution(y, adj)
        y_val, target = torch.max(y, 1)
        y_val, target = y_val.to(self.device), target.to(self.device)
        mask = y_val[self.un_idx] >= self.gamma
        self._training_set_prepare(target, adj)

        # intra-class mixup
        eq_x, eq_y, mixup_adj, eq_idx = self._mixup_intra_class(x, y, adj, self.t_idx, self.un_idx[mask])
        # inter-class mixup
        neq_x_lu, neq_y_lu = self._mixup_inter_class(x, y, adj, self.t_idx, self.un_idx[mask])

        neq_x_ll, neq_y_ll = self._mixup_inter_class(x, y, adj, self.t_idx, self.t_idx)
        neq_x = torch.cat([neq_x_lu, neq_x_ll], dim=0)
        neq_y = torch.cat([neq_y_lu, neq_y_ll], dim=0)

        return eq_x, eq_y, neq_x, neq_y, mixup_adj, self.E, eq_idx

    def _mixup_inter_class(self, x, y, adj, source_idx, target_idx, ll=False):
        lam = self._generate_lam()
        y_val, target = torch.max(y, 1)

        select_un_idx, select_l_idx = self._sampling(target, adj, source_idx, target_idx, eq=False, ll=ll)

        x = lam * x[select_l_idx, :] + (1 - lam) * x[select_un_idx, :]
        y = lam * y[select_l_idx, :] + (1 - lam) * y[select_un_idx, :]

        return x, y

    def _mixup_intra_class(self, x, y, adj, source_idx, target_idx):
        lam = self._generate_lam()
        y_val, target = torch.max(y, 1)

        select_un_idx, select_l_idx = self._sampling(target, adj, source_idx, target_idx, eq=True)
        x[select_l_idx, :] = lam * x[select_l_idx, :] + (1 - lam) * x[select_un_idx, :]
        y[select_l_idx, :] = lam * y[select_l_idx, :] + (1 - lam) * y[select_un_idx, :]

        mixup_adj = adj
        num_nodes = select_un_idx.shape[0]
        if num_nodes > 0:
            # for sparse version
            row_and_col = [
                (
                    torch.ones_like(adj[1][adj[0] == select_un_idx[index]]) * select_l_idx[index],
                    adj[1][adj[0] == select_un_idx[index]]
                )
                for index in range(num_nodes)
            ]
            new_row = torch.cat([item[0] for item in row_and_col])
            new_col = torch.cat([item[1] for item in row_and_col])

            mixup_adj = torch.cat((adj, torch.stack([new_row, new_col], dim=0)), dim=1)

            # for dense version
            # adj = to_dense_adj(adj).squeeze(0)
            # row = lam * adj[select_l_idx, :] + (1 - lam) * adj[select_un_idx, :]
            # col = lam * adj[:, select_l_idx] + (1 - lam) * adj[:, select_un_idx]
            # adj[select_l_idx, :], adj[:, select_l_idx] = row, col

        return x, y, mixup_adj, select_l_idx

    def _calculate_sampling_weight(self, nb_v_similarity, degree, eq=True):
        # size = nb_v_similarity.shape[1]
        # block_size = int(size / 5)
        if eq:
            #  For same-class Mixup, nb_v_similarity should be monotonically increasing
            weight = torch.exp(self.beta_s * nb_v_similarity) / (1 + self.beta_d * degree)
        else:
            # For different-class Mixup, nb_v_similarity should be monotonically decreasing
            weight = torch.exp(-self.beta_s * nb_v_similarity) / (1 + self.beta_d * degree)

        return weight

    def _sampling(self, target, adj, source_idx, target_idx, eq=True, ll=False):
        target_idx_list = []
        source_idx_list = []
        deg = degree(adj[0])
        for i in range(self.nclass):
            sub_source_idx = source_idx[target[source_idx] == i]
            sub_target_idx = target_idx[target[target_idx] == i] if eq else target_idx[target[target_idx] != i]
            target_size = sub_target_idx.shape[0]
            source_size = sub_source_idx.shape[0]
            size = target_size if target_size < source_size else source_size

            if size > 0:
                source_nb_dist = self.nb_dist[sub_source_idx[:size]]
                target_nb_dist = self.nb_dist[sub_target_idx]
                similarity_matrix = cosine_similarity_matrix(source_nb_dist, target_nb_dist)
                weights = torch.zeros_like(similarity_matrix).to(self.device)
                deg_mat = deg[sub_target_idx]  # .expand(similarity_matrix.shape[0], -1)
                weights = self._calculate_sampling_weight(nb_v_similarity=similarity_matrix, degree=deg_mat, eq=eq)
                norm_weights = weights.softmax(-1)
                indices = torch.multinomial(norm_weights, num_samples=1, replacement=False)
                if indices.dim() > 1:
                    indices = indices.squeeze(1)
                select_target_idx = indices
            else:
                select_target_idx = sub_target_idx[:size]
            target_idx_list.append(select_target_idx)
            source_idx_list.append(sub_source_idx[:size])
        target_idx = torch.cat(target_idx_list, dim=0)
        source_idx = torch.cat(source_idx_list, dim=0)

        return target_idx, source_idx
