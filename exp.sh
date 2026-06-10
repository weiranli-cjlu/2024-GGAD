python run.py --dataset ACM --num_trials 10 \
	--embedding_dim 64 \
	--readout min \
	--lr 0.009488338054951556 \
	--weight_decay 0.001 \
	--negsamp_ratio 4 \
	--confidence_margin 1.918880332265193 \
	--normal_rate 0.9994151465153001 \
	--outlier_rate 0.28396109709804696 \
	--mean 0.09894443422011567 \
	--var 0.06864221091515169 \
	--num_epoch 200

python run.py --dataset cora --num_trials 10 \
	--embedding_dim 128 \
	--readout avg \
	--lr 9.425167673953773e-05 \
	--weight_decay 0.001 \
	--negsamp_ratio 1 \
	--confidence_margin 0.2984927262666409 \
	--normal_rate 0.6358501353143552 \
	--outlier_rate 0.18402040139329565 \
	--mean 0.02594187457990358 \
	--var 0.06038086841490703 \
	--num_epoch 50

python run.py --dataset YelpChi --num_trials 10 \
	--embedding_dim 128 \
	--readout min \
	--lr 0.000153658234283706 \
	--weight_decay 1e-05 \
	--negsamp_ratio 2 \
	--confidence_margin 1.3258283017779549 \
	--normal_rate 0.2569297861044923 \
	--outlier_rate 0.21037194404971513 \
	--mean 0.03867353463005374 \
	--var 0.09367299887367346 \
	--num_epoch 400

python run.py --dataset Flickr --num_trials 10 \
	--embedding_dim 128 \
	--readout min \
	--lr 0.00668495254327513 \
	--weight_decay 0.001 \
	--negsamp_ratio 3 \
	--confidence_margin 0.999672557542712 \
	--normal_rate 0.5249588108958066 \
	--outlier_rate 0.19724456373001475 \
	--mean 0.06100283203921914 \
	--var 0.02841895354344235 \
	--num_epoch 200

python run.py --dataset twitter --num_trials 10 \
	--embedding_dim 128 \
	--readout min \
	--lr 0.0016730402817820241 \
	--weight_decay 1e-06 \
	--negsamp_ratio 1 \
	--confidence_margin 0.6780602616231216 \
	--normal_rate 0.5722807884690141 \
	--outlier_rate 0.13526405540621358 \
	--mean 0.029122914019804193 \
	--var 0.06118528947223795 \
	--num_epoch 400

python run.py --dataset BlogCatalog --num_trials 10 \
	--embedding_dim 64 \
	--readout min \
	--lr 1.2715708894220005e-05 \
	--weight_decay 0.001 \
	--negsamp_ratio 4 \
	--confidence_margin 0.9946703928732321 \
	--normal_rate 0.9803599967502249 \
	--outlier_rate 0.2990613272783179 \
	--mean 0.042598383771975314 \
	--var 0.007209192218350332 \
	--num_epoch 50

python run.py --dataset tolokers --num_trials 10 \
	--embedding_dim 64 \
	--readout max \
	--lr 0.004295312362786773 \
	--weight_decay 0.0 \
	--negsamp_ratio 2 \
	--confidence_margin 0.17743166972153818 \
	--normal_rate 0.7404480702303398 \
	--outlier_rate 0.2246127126825752 \
	--mean 0.09556679250641067 \
	--var 0.005702256261760205 \
	--num_epoch 50