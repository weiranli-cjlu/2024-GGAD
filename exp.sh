python run.py --dataset ACM --num_trials 10 --use_best \
	--embedding_dim 512 \
	--readout avg \
	--lr 0.021206502513245407 \
	--weight_decay 1e-05 \
	--negsamp_ratio 2 \
	--confidence_margin 1.9243590578082754 \
	--normal_rate 0.784099192041822 \
	--outlier_rate 0.1096307747915124 \
	--mean 0.09934176073551038 \
	--var 0.06915293964779148 \
	--num_epoch 100

python run.py --dataset cora --num_trials 10 --use_best \
	--embedding_dim 512 \
	--readout avg \
	--lr 0.01329980987940469 \
	--weight_decay 1e-08 \
	--negsamp_ratio 3 \
	--confidence_margin 0.8143819234869726 \
	--normal_rate 0.6304632410332233 \
	--outlier_rate 0.18009220771137674 \
	--mean 0.04802256725429582 \
	--var 0.09981398701113815 \
	--num_epoch 400

python run.py --dataset YelpChi --num_trials 10 --use_best \
	--embedding_dim 512 \
	--readout max \
	--lr 0.0036217766119844177 \
	--weight_decay 0.0001 \
	--negsamp_ratio 5 \
	--confidence_margin 1.7899102573552623 \
	--normal_rate 0.7612735316299505 \
	--outlier_rate 0.07768296915763115 \
	--mean 0.04736454359047703 \
	--var 0.0885338867958875 \
	--num_epoch 50

python run.py --dataset Flickr --num_trials 10 --use_best \
	--embedding_dim 512 \
	--readout min \
	--lr 0.017922185469500057 \
	--weight_decay 1e-06 \
	--negsamp_ratio 1 \
	--confidence_margin 1.7507330290067844 \
	--normal_rate 0.7905900447390287 \
	--outlier_rate 0.07987453731713783 \
	--mean 0.034389549075890015 \
	--var 0.049896092498296886 \
	--num_epoch 100

python run.py --dataset twitter --num_trials 10 --use_best \
	--embedding_dim 256 \
	--readout avg \
	--lr 0.0001773488562686114 \
	--weight_decay 0.001 \
	--negsamp_ratio 5 \
	--confidence_margin 0.5783863620681918 \
	--normal_rate 0.547523655303147 \
	--outlier_rate 0.09725470984686319 \
	--mean 0.028484049437746763 \
	--var 0.0036886947354532796 \
	--num_epoch 50

python run.py --dataset BlogCatalog --num_trials 10 --use_best \
	--embedding_dim 300 \
	--readout min \
	--lr 0.017589437523721014 \
	--weight_decay 1e-06 \
	--negsamp_ratio 1 \
	--confidence_margin 1.804709940111321 \
	--normal_rate 0.9103762514469974 \
	--outlier_rate 0.1935994226092477 \
	--mean 0.03390297910487007 \
	--var 0.03492095746126609 \
	--num_epoch 100

python run.py --dataset tolokers --num_trials 10 --use_best \
	--embedding_dim 300 \
	--readout min \
	--lr 0.017589437523721014 \
	--weight_decay 1e-06 \
	--negsamp_ratio 1 \
	--confidence_margin 1.804709940111321 \
	--normal_rate 0.9103762514469974 \
	--outlier_rate 0.1935994226092477 \
	--mean 0.03390297910487007 \
	--var 0.03492095746126609 \
	--num_epoch 100