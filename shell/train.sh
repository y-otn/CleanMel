cd ../

gpus=$1

python -m model.CleanMelTrainer_mapping fit \
    --config ./configs/model/cleanmel.yaml \
    --config ./configs/dataset/train.yaml \
    --trainer.devices=${gpus} \
    --data.batch_size=[8,20] \
    --data.dataset_len=[100000,3000,3000] \
    --model.exp_name ./CleanMel_S_mapping/ \
    --model.vocos_ckpt  ./pretrained/separate_models/vocos/vocos_offline.pt \
    --model.vocos_config ./configs/model/vocos_offline.yaml \