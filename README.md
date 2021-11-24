# SwisE
Switch spaces for knowledge graph embeddings.

```
Requirements:
python3
pytorch
numpy
tqdm
```


To reproduce the reported results, please follow the following steps:
1. Run "source set_env.sh"

2. Run the following commands for WN18RR
CUDA_VISIBLE_DEVICES=YourDeviceID python3 run.py --model=SwisE --max_epochs=100 --optimizer=Adam --regularizer=N3 --multi_c --rank=100  --batch_size 500 
--neg_sample_size 50 --init_size 0.001 --learning_rate 0.01  --bias learn --valid 1 --dataset=WN18RR 
--k=2 -manifolds Spherical Spherical Spherical Spherical  Euclidean

3. Run the following commands for FB15K237
CUDA_VISIBLE_DEVICES=YourDeviceID python3 run.py --model=SwisE --max_epochs=100 --optimizer=Adam --regularizer=N3 --multi_c --rank=100 --batch_size 500 
--neg_sample_size 50 --init_size 0.001 --learning_rate 0.005 --bias learn --valid 1 --dataset=FB237 
--k=4 -manifolds Hyperbolic Hyperbolic Hyperbolic Spherical Spherical

```
@article{zhang2021switch,
  title={Switch spaces: Learning product spaces with sparse gating},
  author={Zhang, Shuai and Tay, Yi and Jiang, Wenqi and Juan, Da-cheng and Zhang, Ce},
  journal={arXiv preprint arXiv:2102.08688},
  year={2021}
}
```

