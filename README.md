# CleanMel
Pytorch implementation of "CleanMel: Mel-Spectrogram Enhancement for Improving Both Speech Quality and ASR".

[Paper :star_struck:](https://arxiv.org/abs/2502.20040) **|** [Demos :notes:](https://audio.westlake.edu.cn/Research/CleanMel.html) **|** [Issues :sweat_smile:](https://github.com/Audio-WestlakeU/CleanMel/issues)
 **|** [Lab :hear_no_evil:](https://github.com/Audio-WestlakeU) **|** [Contact :kissing_heart:](https://saoyear.github.io)

## Noticement
- The offline-CleanMel-S-map/mask and online-CleanMel-S-map checkpoints are available now.

## Introduction

<p align="center">
  <img src="./src/imgs/cleanmel_arch.png" width="800" />
</p>

We introduce CleanMel, a Mel-spectrogram enhancement method which generates enhanced (denoise + dereverberation) logMel spectrograms. The output of CleanMel could be used for Vocoder or ASR systems to obtain enhanced waveforms or transcriptions.

## Environment

Create a new conda environment for CleanMel:

```
conda create -n CleanMel python==3.10.14
```
Setup the required packages:
```
pip install -r requirements.txt
```
<font color=gray> Hint: If you have any problem w.r.t. the environments. Please first make sure your `mamba_ssm`, `torch` and `pytorch-lightning` versions are identical to those in the `requirements.txt`.</font>


## Pretrained checkpoints

**Enhancement model**

We provides the 3 pretrained checkpoints by default in `./pretrained/enhancement`:
- offline_CleanMel_S_map
- offline_CleanMel_S_mask
- online_CleanMel_S_map

offline_CleanMel_L_map and offline_CleanMel_L_mask will be uploaded via cloud drive since their large sizes.

**Vocos Model**

The pretrained offline/online vocos model checkpoints can be found [here](https://drive.google.com/file/d/13Q0995DmOLMQWP-8MkUUV9bJtUywBzCy/view?usp=drive_link). Please copy the downloaded vocos checkpoints to `./pretrained/vocos`.

`./pretrained` **folder structure**

The `pretrained` folder should be structured as follows:
```
pretrained
├── enhancement
|   ├── offline_CleanMel_S_map.ckpt
|   ├── offline_CleanMel_S_mask.ckpt
|   |── online_CleanMel_S_map.ckpt
|   |── ...
├── vocos
|   ├── vocos_offline.pt
|   |── vocos_online.pt
```


## Inference

**Infernece with Pretrained Models**

The `inference.sh` in `./shell` folder provides script for model inference, the input arguments are (in order):
1. `<GPU_ID>` : the GPU ID for inference;
2. `<Mode>` : `online` or `offline`;
3. `<Model size>` : `S` or `L`;
4. `<Model output>` : `map` or `mask`.

E.g., to inference with `offline_CleanMel_S_map` on `GPU:0`:
```
cd shell
bash inference 0, offline S map
```
<font color=gray, size=1>Hint: the number behind indicates your GPU ID. `0,` means using only `GPU:0` for inference. And change it value to `0,1,...` to allow more `GPUS` for inference.</font>

**Sanity Check** 

To check if you are running CleanMel correctly, an inference example of `noisy_CHIME-real_F05_442C020S_STR_REAL.wav` is provided in `./src/inference_examples/` folder.

**Customize Inference**

By default, the script would inference the noisy waveforms in `./src/demos/` and save the enhanced waveforms to `./my_output` folder. 

To **inference with your own data**, please change the `speech_folder` argument in the script. To customize the output directory, please change the `output_folder` argument in the script. 

## Training
To train with your own data, please modify the `./config/dataset/train.yaml` file, and change the `speech_dir`, `noise_dir` and `rir_dir` to your own data path.

The `train.sh` in `./shell` folder provides script for model training, the input arguments are (in order):
1. `<GPU_ID>` : the GPU ID for training;
2. `<Mode>` : `online` or `offline`;    
3. `<Model size>` : `S` or `L`;
4. `<Model output>` : `map` or `mask`.

E.g., to train with `offline_CleanMel_L_mask` on `GPU:0`:
```
cd shell
bash train 0, offline L mask
```

## Performance

### Speech enhancement performance

The speech enhancement peroformance is evaluated by DNSMOS and PESQ on several datasets. The results are shown in the following:
<p align="center">
  <img src="./src/imgs/dnsmos_performance.png" width="800" />
</p>

<p align="center">
  <img src="./src/imgs/pesq_performance.png" width="800" />
</p>

### ASR performance
We test CleanMel on 3 datasets: CHiME4, REVERB, RealMAN to evaluate its peroformance on English and Chinese ASR tasks. The results are shown in the following:
<p align="center">
  <img src="./src/imgs/asr_performance.png" width="800" />
</p>

Please check the [`asr_infer` branch](https://github.com/Audio-WestlakeU/CleanMel/tree/asr_infer) for the ASR inference details and implementations.

## Citation
```
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

## Reference

The pytorch-lightning structure in this repository is referred from [NBSS](https://github.com/Audio-WestlakeU/NBSS).