python3 tools/train.py \
       --config configs/ppmattingv2/ppmattingv2-stdc1-bgsdata.yml \
       --do_eval \
       --use_vdl --num_workers 20 --save_dir data/PPmattingv2Output/ \
       --save_interval 5000 #--resume_model /data/ModNetOutput/iter_40000