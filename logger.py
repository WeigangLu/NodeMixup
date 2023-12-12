import numpy as np
import os
import os.path as osp


class Logger(object):
    def __init__(self, runs, args=None):
        self.results = [0 for _ in range(runs)]
        self.best_test_results = [0 for _ in range(runs)]
        self.args = args
        self.content = ""
        self.additional_content = ""

        root_path = args.root_path
        self.strategy = "None"
        if args.lam_inter > 0. or args.lam_intra > 0.:
            self.strategy = "NodeMixup"
        self.folder_name = osp.join(root_path, "./results/{}/{}".format(args.dataset,
                                                                        args.train_size if args.train_size > 0 else 'public'))
        self.acc_mean = 0.
        self.acc_std = 0.

    def add_result(self, run, result, best_test_acc):
        assert 0 <= run < len(self.results)
        assert 0 <= run < len(self.best_test_results)
        self.results[run] = result * 100
        self.best_test_results[run] = best_test_acc * 100

    def dump_parameters(self ):
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        filename = os.path.join(self.folder_name, self.args.model + f"-{self.strategy}.txt")
        with open(filename, "a+") as f:
            f.write(self.content + "\t")
            f.write(self.additional_content + "\t")
            f.write(str(self.args) + "\n")

    def add_content(self, content):
        self.additional_content += content

    def print_statistics(self):
        acc_best = np.max(self.results)
        acc_mean = np.mean(self.results)
        acc_std = np.std(self.results)

        best_acc_mean = np.mean(self.best_test_results)
        best_acc_std = np.std(self.best_test_results)

        self.acc_mean = acc_mean
        self.acc_std = acc_std

        print(
            f'{self.args.model} on dataset {self.args.dataset}, with {self.args.train_size} training samples per class, in {len(self.results)} repeated experiment:')
        self.content = f'Final Test: {acc_mean:.2f} ± {acc_std:.2f}\tBest Test: {best_acc_mean:.2f} ± {best_acc_std:.2f}'
        print(self.content)
        self.dump_parameters()
