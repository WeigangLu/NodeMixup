import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv, SAGEConv, SGConv, APPNP

class GCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN, self).__init__()
        self.nlayer = args.nlayer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(dataset.num_features, args.hid_dim))
        for _ in range(self.nlayer - 2):
            self.convs.append(GCNConv(args.hid_dim, args.hid_dim))
        self.convs.append(GCNConv(args.hid_dim, dataset.num_classes))
        self.dropout = args.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        x = F.relu(self.convs[0](x, adj, ))
        for conv in self.convs[1:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            conv_x = conv(x, adj, )
            x = F.relu(conv_x)
        x = self.convs[-1](x, adj, )
        return x


class GAT(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT, self).__init__()
        self.nlayer = args.nlayer
        self.convs = torch.nn.ModuleList()

        self.convs.append(
            GATConv(
                dataset.num_features,
                args.hid_dim,
                heads=args.heads,
                dropout=args.dropout)
        )
        for _ in range(self.nlayer - 2):
            self.convs.append(
                GATConv(
                    args.hid_dim * args.heads,
                    args.hid_dim,
                    heads=args.heads,
                    dropout=args.dropout)
            )

        self.convs.append(
            GATConv(
                args.hid_dim * args.heads,
                dataset.num_classes,
                heads=args.output_heads,
                concat=False,
                dropout=args.dropout)
        )
        self.dropout = args.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj,):
        x = F.elu(self.convs[0](x, adj))
        for conv in self.convs[1:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            conv_x = conv(x, adj)
            x = F.elu(conv_x)
        x = self.convs[-1](x, adj)
        return x


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hid_dim)
        self.lin2 = Linear(args.hid_dim, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.appnp_alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, adj,):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, adj)
        return x


class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args, name="ChebNet"):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, 64, K=2)
        self.conv2 = ChebConv(64, dataset.num_classes, K=2)
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, adj):
        hid = self.conv1(x, adj)
        x = F.relu(hid)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)

        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, args, name="GraphSAGE"):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, args.hid_dim)
        self.conv2 = SAGEConv(args.hid_dim, dataset.num_classes)
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, adj,):
        hid = self.conv1(x, adj)
        x = F.relu(hid)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return x