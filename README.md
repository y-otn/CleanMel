# CleanMel
[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2502.20040)
[![Demos](https://img.shields.io/badge/🎧-Demos-blue)](https://audio.westlake.edu.cn/Research/CleanMel.html)
[![GitHub Issues](https://img.shields.io/github/issues/Audio-WestlakeU/CleanMel)](https://github.com/Audio-WestlakeU/CleanMel/issues)
[![Contact](https://img.shields.io/badge/💌-Contact-purple)](https://saoyear.github.io)

PyTorch implementation of "CleanMel: Mel-Spectrogram Enhancement for Improving Both Speech Quality and ASR".

## Notice 📢
- `Offline-CleanMel-S-map/mask` and `online-CleanMel-S-map` checkpoints now available
- Large models (`offline_CleanMel_L_*`) are coming soon

## Overview 🚀
<p align="center"><img src="./src/imgs/cleanmel_arch.png"alt="jpg name" width="60%"/></p>

**CleanMel** enhances logMel spectrograms for improved speech quality and ASR performance. Outputs compatible with:
- 🎙️ Vocoders for enhanced waveforms
- 🤖 ASR systems for transcription

## Quick Start ⚡

### Environment Setup
```bash
conda create -n CleanMel python=3.10.14
conda activate CleanMel
pip install -r requirements.txt
```

### Inference
```bash 
# Offline example (offline_CleanMel_S_mask)
cd shell
bash inference.sh 0, offline S mask

# Online example (online_CleanMel_S_map)
bash inference.sh 0, online S map
```
**Custom Input**: Modify `speech_folder` in `inference.sh`

**Output**: Results saved to `output_folder` (default to `./my_output`)

### Training
```bash
# Offline training
cd shell
bash train.sh 0,1,2,3 offline L mask
```
Configure datasets in `./config/dataset/train.yaml`

Default 4 GPUs trained with batch size 32

## Pretrained Models 🧠
```bash
pretrained/
├── enhancement/
│   ├── offline_CleanMel_S_map.ckpt
│   ├── offline_CleanMel_S_mask.ckpt
│   ├── online_CleanMel_S_map.ckpt
|   └── ...
└── vocos/
    ├── vocos_offline.pt
    └── vocos_online.pt
```
**Enhancement**: `offline_CleanMel_L_mask/map.ckpt` are coming soon.

**Vocos**: `vocos_offline.pt` and `vocos_online.pt` are [here](https://drive.google.com/file/d/13Q0995DmOLMQWP-8MkUUV9bJtUywBzCy/view?usp=drive_link).

## Performance 📊
### Speech Enhancement
<p align="center"><img src="./src/imgs/dnsmos_performance.png" alt="jpg name" width="70%"/></p>
<p align="center"><img src="./src/imgs/pesq_performance.png" alt="jpg name" width="40%"/></p>

### ASR Accuracy
<p align="center"><img src="./src/imgs/asr_performance.png" alt="png name" width="40%"/></p>

💡 ASR implementation details in [`asr_infer` branch](https://github.com/Audio-WestlakeU/CleanMel/tree/asr_infer)

## Citation 📝
```bibtex
@misc{shao2025cleanmel,
    title={CleanMel: Mel-Spectrogram Enhancement for Improving Both Speech Quality and ASR}, 
    author={Nian Shao and Rui Zhou and Pengyu Wang and Xian Li and Ying Fang and Yujie Yang and Xiaofei Li},
    year={2025},
    eprint={2502.20040},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url={https://arxiv.org/abs/2502.20040}
}
```

## Acknowledgement 🙏
- Built using [NBSS](https://github.com/Audio-WestlakeU/NBSS) template
- Vocoder implementation from [Vocos](https://github.com/gemelo-ai/vocos)