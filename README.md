# NodeMixup: Tackling Under-reaching for Graph Neural Networks (AAAI2024)

#### Installation

```
pip install -r requirements.txt
```



#### Scripts

```bash
python main.py --lr 0.001 --weight_decay 0.0004 --hid_dim 256  --dropout 0.9 --gamma 0.5 --beta_s 0.5 --beta_d 0.5 --temp 0.1 --nlayer 2 --model 'GCN' --mixup_alpha 0.8 --lam_intra 1.0 --lam_inter 2.0  --train_size -1 --dataset 'cora' --device 0 --runs 10 --epochs 500

python main.py --lr 0.01 --weight_decay 0.0005 --hid_dim 64  --dropout 0.5 --gamma 0.5 --beta_s 2.0 --beta_d 1.5 --temp 0.1 --nlayer 2 --model 'GCN' --mixup_alpha 0.4 --lam_intra 1.0 --lam_inter 1.5  --train_size -1 --dataset 'citeseer' --device 0 --runs 10 --epochs 500

python main.py --lr 0.01 --weight_decay 0.0005 --hid_dim 256  --dropout 0.5 --gamma 0.7 --beta_s 1.0 --beta_d 2.0 --temp 0.1 --nlayer 2 --model 'GCN'  --mixup_alpha 0.4 --lam_intra 1.0 --lam_inter 1.5  --train_size -1 --dataset 'pubmed' --device 0 --runs 10 --epochs 500


```


#### Citation

```
@inproceedings{lu2024nodemixup,
  title={Nodemixup: Tackling under-reaching for graph neural networks},
  author={Lu, Weigang and Guan, Ziyu and Zhao, Wei and Yang, Yaming and Jin, Long},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={13},
  pages={14175--14183},
  year={2024}
}
```

