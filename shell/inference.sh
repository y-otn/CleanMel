cd ../

gpus=$1

# Model configurations 
n_hidden=96
n_layers=8
FFT=512
HOP=128
n_mels=80

python -m model.CleanMelTrainer_mapping predict \
    --config ./configs/model/cleanmel.yaml \
    --config ./configs/dataset/inference.yaml \
    --trainer.devices=${gpus} \
    --model.exp_name ./logs/CleanMel_S_L1/demos/ \
    --model.arch_ckpt ./pretrained/separate_models/enhancement/CleanMel_S_L1.ckpt \
    --model.vocos_ckpt  ./pretrained/separate_models/vocos/vocos_offline.pt \
    --model.vocos_config ./configs/model/vocos_offline.yaml \
    --model.output_path ./my_output/