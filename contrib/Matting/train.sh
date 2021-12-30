export PYTHONPATH="$PYTHONPATH:../../"
python train.py --config configs/modnet/modnet_hrnet_w18.yml --do_eval --use_vdl --save_interval 1000 --save_dir output

# 腾讯云
python train.py --config configs/modnet/modnet_hrnet_w18.yml --do_eval --use_vdl --save_interval 1000 --save_dir /data/ModNetOutput/ --num_workers 5 --resume_model /data/ModNetOutput/iter_40000