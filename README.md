# CleanMel
Pytorch implementation of "CleanMel: Mel-Spectrogram Enhancement for Improving Both Speech Quality and ASR".

[Paper :star_struck:](https://arxiv.org/abs/2502.20040) **|** [Demos :notes:](https://audio.westlake.edu.cn/Research/CleanMel.html) **|** [Issues :sweat_smile:](https://github.com/Audio-WestlakeU/CleanMel/issues)
 **|** [Lab :hear_no_evil:](https://github.com/Audio-WestlakeU) **|** [Contact :kissing_heart:](https://saoyear.github.io)

## Todo List
- [x] A quick inference demo is provided in `shell/inference.sh` | 2025/03/04
- [ ] Complete the README.md | 2025/03
- [ ] Upload the Vocos model | 2025/03
- [ ] Full inference codes + model checkpoints | 2025/03
- [ ] Training codes | 2025/04

## Introduction

## Getting Started
A quick inference demo is provided in `shell/inference.sh`. 
And the model `CleanMel_S_L1.ckpt` is provided in the `pretrained/` folder.
You can run the script to get the enhanced mel-spectrogram.
```bash
cd ./shell
bash inference.sh
```
Currently, the Vocos model is not uploaded yet, so only the enhanced Mel-spectrogram could be obtained. And the enhanced waveform could not be generated.

## Performance

## Citation
