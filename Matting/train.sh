python3 tools/train.py \
       --config configs/ppmattingv2/ppmattingv2-stdc1-bgsdata.yml \
       --do_eval \
       --num_workers 20 --save_dir data/PPmattingv2Output/ \
       --save_interval 10000 --precision fp16 --use_vdl --log_image_iters 99999999 --resume_model data/PPmattingv2Output/iter_1000

# 16fp不支持log_image_iters