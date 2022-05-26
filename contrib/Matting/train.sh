export PYTHONPATH="$PYTHONPATH:../../"
python train.py --config configs/modnet/modnet_hrnet_w18_bgsdata.yml --do_eval --use_vdl --save_interval 1000 --save_dir output --params_preload ./model_old.pdparams --log_iters 100 --num_workers 16

# 腾讯云
python train.py --config configs/modnet/modnet_mobilenetv3_lager.yml --do_eval --use_vdl --save_interval 1000 --save_dir /data/ModNetOutput/ --num_workers 20 --resume_model /data/ModNetOutput/iter_40000

# 预测
python predict.py --config ./configs/modnet/modnet_hrnet_w18_bgsdata.yml  --image_path ../../testimage --save_dir ./output --model_path ./params/model.pdparams

# ai studio 量化
python train.py --config configs/modnet/modnet_hrnet_w18_bgsdata.yml --do_eval --use_vdl --save_interval 1000 --save_dir ./ModnetQuant --params_preload ./src_model/model.pdparams --quant