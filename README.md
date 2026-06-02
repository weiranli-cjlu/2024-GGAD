## Requirements

To install requirements:

```bash
uv venv -p 3.12
uv pip install torch==2.11.0 scikit-learn tqdm optuna pandas --torch-backend=cu128
```

## 调优脚本
```bash
python tune.py \
  --dataset questions \
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

python run.py --dataset twitter \
    --num_trials 10 \
    --lr 0.036691008189209895 --weight_decay 0.0001 \
    --embedding_dim 128 --readout min \
    --negsamp_ratio 1 --mean 0.08700121482468193\
    --var 0.0978618342232764 --confidence_margin 0.13841495513661886 \
    --normal_rate 0.8493578609931441 --outlier_rate 0.23566545777545664

python run.py --dataset tolokers \
    --num_trials 10 \
    --lr 9.419363848421692e-05 --weight_decay 0.001 \
    --embedding_dim 64 --readout avg \
    --negsamp_ratio 4 --mean 0.018563594430595222\
    --var 0.09527916569719447 --confidence_margin 1.758036245350051 \
    --normal_rate 0.3461878313340722 --outlier_rate 0.24143358183464347

python run.py --dataset Amazon \
    --num_trials 10 \
    --lr 0.0018594905297927152 --weight_decay 0.0 \
    --embedding_dim 64 --readout max \
    --negsamp_ratio 4 --mean 0.013069794252316608\
    --var 0.036096315999879905 --confidence_margin 1.2180772843597365 \
    --normal_rate 0.4441505274691648 --outlier_rate 0.20747353804958163

python run.py --dataset ACM \
    --num_trials 10 \
    --lr 0.01177936790365981 --weight_decay 1e-05 \
    --embedding_dim 64 --readout max \
    --negsamp_ratio 2 --mean 0.018098590135138717\
    --var 0.04952336909084007 --confidence_margin 0.13650970878929247 \
    --normal_rate 0.3030734867833197 --outlier_rate 0.03640399456902767

python run.py --dataset Flickr \
    --num_trials 10 \
    --lr 0.0018594905297927152 --weight_decay 0.0 \
    --embedding_dim 64 --readout max \
    --negsamp_ratio 4 --mean 0.013069794252316608\
    --var 0.036096315999879905 --confidence_margin 1.2180772843597365 \
    --normal_rate 0.4441505274691648 --outlier_rate 0.20747353804958163

python run.py --dataset YelpChi \
    --num_trials 10 \
    --lr 0.000782327289426032 --weight_decay 1e-08 \
    --embedding_dim 512 --readout max \
    --negsamp_ratio 1 --mean 0.01323593471688772\
    --var 0.09550611000260706 --confidence_margin 0.6140148733302279 \
    --normal_rate 0.398359550901491 --outlier_rate 0.05811926128765506

python run.py --dataset Reddit \
    --num_trials 10 \
    --lr 0.007852741389047083 --weight_decay 1e-06 \
    --embedding_dim 300 --readout max \
    --negsamp_ratio 5 --mean 0.0503819330325349\
    --var 0.013204222922303095 --confidence_margin 0.928423826810507 \
    --normal_rate 0.9802100898099548 --outlier_rate 0.053643375269955024

python run.py --dataset weibo \
    --num_trials 10 \
    --lr 7.453072195439607e-05 --weight_decay 0.0 \
    --embedding_dim 512 --readout avg \
    --negsamp_ratio 5 --mean 0.09214916846584609\
    --var 0.027337443517333814 --confidence_margin 0.48634207674284113 \
    --normal_rate 0.17769349807820645 --outlier_rate 0.18740438156489012
```