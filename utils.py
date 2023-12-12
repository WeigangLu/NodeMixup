import torch
import torch.nn.functional as F
from torch_geometric.utils import degree


def sharpen(logs, temp):
    val = (torch.pow(logs, 1. / temp) / torch.sum(torch.pow(logs, 1. / temp), dim=1, keepdim=True)).detach()
    val[val.isnan()] = 1e-16
    return val


def Cross_entropy_loss(x, target, mask, weight=None):
    if mask.sum() == 0:
        return 0.
    target = target[mask]
    x = x[mask]
    loss = F.nll_loss(x.log_softmax(-1), target)

    return loss


def Accuracy(x, target, mask):
    x = x[mask]
    target = target[mask]
    x = x.max(1)[1]
    correct = x.eq(target)
    acc = correct.sum().item() / target.squeeze().shape[0]
    false = (1 - correct.float()).bool()
    return acc, false


def centrality(edge_index=None):
    assert edge_index is not None
    deg = degree(edge_index)
    scale_deg = (deg - deg.min()) / (deg.max() - deg.min())
    cen = torch.sigmoid(scale_deg)

    return cen


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def dirichlet_energy(x, adj):
    from torch_geometric.utils import get_laplacian
    L = get_laplacian(adj)
    sp_L = torch.sparse_coo_tensor(indices=L[0], values=L[1], size=(x.shape[0], x.shape[0]))
    inputs = torch.matmul(torch.transpose(x, 1, 0), torch.mm(sp_L, x))
    DG = torch.trace(inputs)

    return DG
