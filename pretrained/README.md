## Pretrained checkpoints

We provide the pretrained checkpoints in two ways:
1. The complete model checkpoint of the pytorch lightning trainer model, including the enhancement model and Vocos neural vocoder.
2. The separated Mel enhancement network and vocos model checkpoints.

### Complete model checkpoint

You could use the complete model checkpoint to inference the noisy waveforms directly in 
```bash
cd shell
bash inference.sh
```

### Separated model checkpoints

We provide the separated model checkpoints to enable the users to try CleanMel in a customized way. E.g., you could place the CleanMel network in the front of your own vocoder model/ASR model, when doing this, you might need the CleanMel checkpoint only, provided in
`./pretrained/separate_models/enhancement/`.