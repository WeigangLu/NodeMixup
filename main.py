import argparse
from train import trail_run
from logger import Logger
import numpy as np
from dataset import load_data


def print_results(epoch, result_dict):
    print(
        f"epoch:{epoch}:\ttrain loss:{result_dict['train_loss']:.2f}, train acc:{result_dict['train_acc']:.4f} | "
        f"val loss:{result_dict['val_loss']:.2f}, val acc:{result_dict['val_acc']:.4f}"
        f" | test loss:{result_dict['test_loss']:.2f}, test acc:{result_dict['test_acc']:.4f}")


def check_results(result_dict):
    for val in result_dict.values():
        if isinstance(val, int) or isinstance(val, float):
            if np.isinf(val) or np.isnan(val):
                exit()


def run(args):
    logger = Logger(runs=args.runs, args=args)

    dataset = load_data(args)

    for i in range(args.runs):
        if args.runs > 1:
            args.seed = i + 1
        best_acc, best_step, final_acc, best_test_acc, best_test_step = 0, 0, 0, 0, 0
        r = trail_run(args, dataset)
        for step in range(args.epochs):
            result_dict, var_dict = next(r)
            check_results(result_dict)
            if result_dict['test_acc'] > best_test_acc:
                best_test_acc = result_dict['test_acc']
                best_test_step = step + 1
            if result_dict['val_acc'] > best_acc:
                best_step = step + 1
                best_acc = result_dict['val_acc']
                final_acc = result_dict['test_acc']
            if args.print:
                print_results(step, result_dict)
        print("")
        print("RUN %d: \t test acc: %.2f at epoch %d" % (i + 1, final_acc * 100, best_step))
        print("RUN %d: \t best test acc: %.2f at epoch %d" % (i + 1, best_test_acc * 100, best_test_step))
        logger.add_result(i, final_acc, best_test_acc)
    logger.add_content(f"# Params = {var_dict['num_paras']}")
    logger.print_statistics()

    return result_dict, var_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general hyperparameters
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay rate on parameters")
    parser.add_argument('--hid_dim', type=int, default=64, help="hidden dimensionality of node representations")
    parser.add_argument('--dropout', type=float, default=0.5)

    # specific for NodeMixup
    parser.add_argument('--gamma', type=float, default=0.3, help="threshold for pseudo labeling")
    parser.add_argument('--beta_s', type=float, default=1, help="tuning strength of NLP similarity in NLD-aware sampling")
    parser.add_argument('--beta_d', type=float, default=1, help="tuning strength of node degree in NLD-aware sampling")
    parser.add_argument('--temp', type=float, default=0.1, help="sharpness of distribution")
    parser.add_argument('--mixup_alpha', type=float, default=0.8, help="determing the Beta distribution")
    parser.add_argument('--lam_intra', type=float, default=1, help="balance hyperparameter of intra-class mixup loss")
    parser.add_argument('--lam_inter', type=float, default=1, help="balance hyperparameter of inter-class mixup loss")

    # for backbone model
    parser.add_argument('--nlayer', default=2, type=int, help="number of layers")
    parser.add_argument('--model', type=str, default='GCN', help="backbone model")
    parser.add_argument('--heads', type=int, default=8, help="number of GAT attention hidden heads")
    parser.add_argument('--output_heads', default=1, type=int, help="number of GAT attention output heads")
    parser.add_argument('--appnp_alpha', type=float, default=0.4, help="teleport probaility for APPNP")
    
    # for experimental setting
    parser.add_argument('--train_size', type=int, default=-1, help="number of samples per class")
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--root_path', type=str, default='XX/XX/XX/NodeMixup')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)

    args = parser.parse_args()
    run(args)
