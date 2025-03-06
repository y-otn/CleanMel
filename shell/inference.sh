cd ../

gpus=$1

python -m model.CleanMelTrainer_mapping predict \
    --config ./configs/model/cleanmel.yaml \
    --config ./configs/dataset/inference.yaml \
    --trainer.devices=${gpus} \
    --model.exp_name ./logs/CleanMel_S_L1/demos/ \
    --model.arch_ckpt ./pretrained/separate_models/enhancement/CleanMel_S_L1.ckpt \
    --model.vocos_ckpt  ./pretrained/separate_models/vocos/vocos_offline.pt \
    --model.vocos_config ./configs/model/vocos_offline.yaml \
    --dataset.speech_dir ./src/demos/ \
    --model.output_path ./my_output/