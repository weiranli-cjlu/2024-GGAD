## Requirements

To install requirements:

```bash
uv venv -p 3.12
uv pip install torch==2.11.0 scikit-learn tqdm optuna pandas --torch-backend=cu128
```

## 调优脚本
```bash
python tune.py \
  --dataset twitter \
  --data_dir ~/datasets/GAD/mat \
  --n_trials 20 \
  --num_trials 1 \
  --metric auc \
  --storage sqlite:///results/ggad_optuna.db
```

## 实验脚本
```bash
python run.py --dataset Disney \
    --num_trials 10 \
    --lr 2.941910265039074e-05 --weight_decay 1e-05 \
    --embedding_dim 300 --readout max \
    --negsamp_ratio 3 --mean 0.07522929863728513\
    --var 0.019821247342103117 --confidence_margin 1.6406293866892558 \
    --normal_rate 0.6009260726496264 --outlier_rate 0.24211369627783863

python run.py --dataset book \
    --num_trials 10 \
    --lr 0.005019951100475927 --weight_decay 0.0001 \
    --embedding_dim 512 --readout avg \
    --negsamp_ratio 3 --mean 0.04344166255581208\
    --var 0.03117958819941026 --confidence_margin 0.5320877931493141 \
    --normal_rate 0.32892083359335367 --outlier_rate 0.02682845649392393

python run.py --dataset Facebook \
    --num_trials 10 \
    --lr 0.00020691281319936292 --weight_decay 0.0001 \
    --embedding_dim 256 --readout max \
    --negsamp_ratio 1 --mean 0.008858301979005723\
    --var 0.05368362913746623 --confidence_margin 1.0063801551525478 \
    --normal_rate 0.33906019241012575 --outlier_rate 0.125095829726733

python run.py --dataset cora \
    --num_trials 10 \
    --lr 2.0813423943167893e-05 --weight_decay 1e-08 \
    --embedding_dim 128 --readout max \
    --negsamp_ratio 2 --mean 0.022270888084003158\
    --var 0.05752326595148801 --confidence_margin 0.556152869142903 \
    --normal_rate 0.2912580111273405 --outlier_rate 0.06695350741203251

python run.py --dataset citeseer \
    --num_trials 10 \
    --lr 2.7143950040714688e-05 --weight_decay 0.0001 \
    --embedding_dim 64 --readout min \
    --negsamp_ratio 4 --mean 0.04579212338568714\
    --var 0.005354053867416465 --confidence_margin 0.7293467521346132 \
    --normal_rate 0.3818653710622053 --outlier_rate 0.2980248354236761
```