export PYTHONPATH="$PYTHONPATH:../../"
python train.py --config configs/modnet/modnet_hrnet_w18.yml --do_eval --use_vdl --save_interval 1000 --save_dir output