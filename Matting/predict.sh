python tools/predict.py --config configs/ppmattingv2/ppmattingv2-stdc1-bgsdata.yml --model_path params/model.pdparams --image_path test_images --save_dir ./output/results --fg_estimate False

python tools/predict.py --config configs/modnet/modnet_hrnet_w18_bgsdata.yml --model_path params/modnet.pdparams --image_path test_images --save_dir ./output/results --fg_estimate False --add_opt SaveOnnx

python tools/predict.py --config configs/ppmatting/ppmatting-hrnet_w18-human_512.yml --model_path params/ppmatting-hrnet_w18-human_512.pdparams --image_path test_images --save_dir ./output/results --fg_estimate False