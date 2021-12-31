export PYTHONPATH="$PYTHONPATH:../../"
python train.py --config configs/modnet/modnet_hrnet_w18.yml --do_eval --use_vdl --save_interval 1000 --save_dir output

# 腾讯云
python train.py --config configs/modnet/modnet_mobilenetv3_lager.yml --do_eval --use_vdl --save_interval 1000 --save_dir /data/ModNetOutput/ --num_workers 20 --resume_model /data/ModNetOutput/iter_40000

# 预测
python predict.py --config .\configs\modnet\modnet_hrnet_w18.yml  --image_path ..\..\testimage\ --save_dir ./output/results --model_path .\params\model.pdparams