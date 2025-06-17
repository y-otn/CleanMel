## Pretrained checkpoints

The checkpoint of pretrained CleanMel models are available in the hugging-face page: https://huggingface.co/WestlakeAudioLab/CleanMel.

### Local download

If you prefer to download the pretrained models locally, please move the checkpoints into the `./pretrained/enhancement/` and `./pretrained/vocos/` directories, respectively. The directory structure should look like this:
```
pretrained/
├── enhancement/
│   ├── offline_CleanMel_S_map.ckpt
│   ├── offline_CleanMel_S_mask.ckpt
│   ├── offline_CleanMel_L_map.ckpt
│   ├── offline_CleanMel_L_mask.ckpt
|   ├── online_CleanMel_S_map.ckpt
│   └── online_CleanMel_S_mask.ckpt
└── vocos/
    ├── vocos_offline.pt
    └── vocos_online.pt  
```

### Inference commands

You could checkout the `inference.sh` script for the inference commands. Here are some examples:
```bash
# FORMAT
# bash inference.sh <gpu_ids>, <online/offline> <S/L> <mask/map> [huggingface]

# ---------------------------------------
# Inference with pretrained models from huggingface

# Offline example (offline_CleanMel_S_mask)
cd shell
bash inference.sh 0, offline S mask huggingface
# Online example (online_CleanMel_S_map)
bash inference.sh 0, online S map huggingface   

# ---------------------------------------
# Inference with local models

# Offline example (offline_CleanMel_S_mask)
cd shell
bash inference.sh 0, offline S mask
# Online example (online_CleanMel_S_map)
bash inference.sh 0, online S map
```