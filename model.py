from encoders import *
import torch
from torch_geometric.utils import dropout_adj


class BaselineModels(torch.nn.Module):
    def __init__(self, dataset, model, args, name=""):
        super(BaselineModels, self).__init__()
        self.dataset = dataset
        layer_dict = {
            "GCN": GCN,
            "GraphSAGE": GraphSAGE,
            "SGC": SGC,
            "ChebNet": ChebNet,
            "GAT": GAT,
            "APPNP": APPNP_Net
        }
        self.model = layer_dict[model](dataset=dataset, args=args)
        self.name = model

    def forward(self, x, adj, mixup_dict=None):
        logits = self.model(x, adj)
        eq_mixup_logits = None
        neq_mixup_logits = None
        if mixup_dict is not None:
            eq_mixup_x, neq_mixup_x, mixup_adj = mixup_dict['eq_mixup_x'], mixup_dict['neq_mixup_x'], mixup_dict['mixup_adj']
            if eq_mixup_x.shape[0] > 0:
                # eq_mixup_logits = self.model(eq_mixup_x, adj=mixup_adj[0], edge_weight=mixup_adj[1])
                eq_mixup_logits = self.model(eq_mixup_x, adj=mixup_adj)
            if neq_mixup_x.shape[0] > 0:
                neq_mixup_logits = self.model(neq_mixup_x, adj=mixup_dict['E'])
        return logits, eq_mixup_logits, neq_mixup_logits

    def get_embeddings(self):
        return self.model.hid_list
