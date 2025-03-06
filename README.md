# CleanMel
Pytorch implementation of "CleanMel: Mel-Spectrogram Enhancement for Improving Both Speech Quality and ASR".

[Paper :star_struck:](https://arxiv.org/abs/2502.20040) **|** [Demos :notes:](https://audio.westlake.edu.cn/Research/CleanMel.html) **|** [Issues :sweat_smile:](https://github.com/Audio-WestlakeU/CleanMel/issues)
 **|** [Lab :hear_no_evil:](https://github.com/Audio-WestlakeU) **|** [Contact :kissing_heart:](https://saoyear.github.io)

## Noticement
- [x] A quick inference demo is provided in `shell/inference.sh`.
- [x] For the ASR inference codes, we released in the `asr_infer` branch.
- [ ] This repo is under development and will be formally released ASAP.


## Introduction


## Getting Started
**Step 1. Pretrained checkpoint:** Download the [pretrained checkpoints](https://drive.google.com/file/d/13Q0995DmOLMQWP-8MkUUV9bJtUywBzCy/view?usp=drive_link) here (Google drive).

**Step 2. Environment configurations:** You might need to create a new conda environment to run CleanMel, to do so:

```
conda create -n CleanMel python==3.10.14
```
To setup the required packages:
```
pip install -r requirements.txt
```
Hint: If you have any problem w.r.t. the environments. Please first make sure your `mamba_ssm`, `torch` and `pytorch-lightning` versions are identical to those in the `requirements.txt`.

## Inference
**Change pretrained checkpoint path:**

In `./shell/inference.sh`, change the `model.arch_ckpt` to `YOUR_PATH/CleanMel_S_L1.ckpt` and `model.vocos_ckpt` to `YOUR_PATH/vocos_offline.pt`.

**Inference provided demos:** 

Using the `inference.sh` in `shell` folder:
```
cd shell
bash inference 0,
```
Hint: the number behind indicates your GPU ID. `0,` means using only `GPU:0` for inference. And change it value to `0,1,...` to allow more `GPUS` for inference.

**Inference your own file:** 

In `inference.sh`, change the `dataset.speech_dir` to your raw waveforms and and `model.output_path` direct to your output folder. Run 
```
bash inference 0,
```
You could also delete `model.output_path`, by default, the waveform will output to the same folder of your pretrained checkpoint.

## Training

## Performance

### Speech enhancement performance

### ASR performance

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