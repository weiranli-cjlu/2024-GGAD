# GGAD 实验命令说明

本文件把原作者 README 中的实验配置整理为命令行格式，便于复现实验与批量多 trial 测试。

## 数据集路径

默认数据路径已改为：

```bash
~/datasets/GAD/mat
```

目录下支持的文件示例：`Amazon.mat`、`Reddit.mat`、`elliptic.mat`、`photo.mat`、`Facebook.mat`、`twitter.mat`、`weibo.mat` 等。命令中的 `--dataset` 会自动尝试大小写匹配。

## 通用参数

```bash
python run.py \
  --dataset Reddit \
  --data_dir ~/datasets/GAD/mat \
  --result_csv results/ggad_results.csv \
  --device auto \
  --num_trials 5 \
  --seed 0
```

也可以显式指定随机种子：

```bash
python run.py --dataset Reddit --device cuda:0 --seeds 0,1,2,3,4
```

训练过程中不打印 epoch 日志，只在所有 trial 完成后输出一次总结，并将结果追加写入 CSV。

## GPU / CPU 设备参数

新增 `--device` 参数：

```bash
# 自动选择：有 CUDA 时使用 cuda:0，否则使用 cpu
python run.py --dataset Reddit --device auto

# 指定第一张 GPU
python run.py --dataset Reddit --device cuda:0

# 强制使用 CPU
python run.py --dataset Reddit --device cpu
```

代码会将 `features`、`adj`、`raw_adj`、`labels`、模型参数、BCE 的 `pos_weight` 和训练标签 `lbl` 统一移动到同一设备。`model.py` 中生成扰动噪声已改为 `torch.randn_like(emb_abnormal)`，避免 GPU 训练时出现 CPU/CUDA 张量不一致错误。

## 原作者实验配置的命令行版本

原作者 README 中说明：局部亲和度 margin `α=0.7`；Photo 和 Reddit 的扰动均值/方差为 `0.02/0.01`；其他大数据集扰动为 `0/0`；生成 outlier 节点比例 Amazon 为 `5%`，其他数据集为 `15%`。

### Reddit

```bash
python run.py \
  --dataset Reddit \
  --num_epoch 300 \
  --lr 1e-3 \
  --confidence_margin 0.7 \
  --mean 0.02 \
  --var 0.01 \
  --outlier_rate 0.15 \
  --device auto \
  --num_trials 5 \
  --seed 0
```

### Photo

```bash
python run.py \
  --dataset photo \
  --num_epoch 100 \
  --lr 1e-3 \
  --confidence_margin 0.7 \
  --mean 0.02 \
  --var 0.01 \
  --outlier_rate 0.15 \
  --device auto \
  --num_trials 5 \
  --seed 0
```

### Elliptic

```bash
python run.py \
  --dataset elliptic \
  --num_epoch 150 \
  --lr 1e-3 \
  --confidence_margin 0.7 \
  --mean 0.0 \
  --var 0.0 \
  --outlier_rate 0.15 \
  --device auto \
  --num_trials 5 \
  --seed 0
```

### Amazon

```bash
python run.py \
  --dataset Amazon \
  --num_epoch 800 \
  --lr 1e-3 \
  --confidence_margin 0.7 \
  --mean 0.0 \
  --var 0.0 \
  --outlier_rate 0.05 \
  --device auto \
  --num_trials 5 \
  --seed 0
```

## CSV 输出字段

`results/ggad_results.csv` 包含以下结果字段，不包含实验配置超参数：

- `time`：实验结束时间，精确到分钟；
- `dataset`：数据集；
- `trial`：trial；
- `auc`、`auprc`：多 trial 统计，格式为 `均值±方差/标准差(最大值)`，例如 `90.21±2.33(91.00)`；
- `train_time_min`：训练耗时，单位分钟；
- `device`：实际使用设备，例如 `cuda:0` 或 `cpu`。

> 注：代码中按常见实验汇报方式使用标准差 `std`，不是方差 `variance`；如需严格输出方差，把 `metric_summary()` 中的 `arr.std(ddof=0)` 改为 `arr.var(ddof=0)` 即可。
