huggingface-cli login


## Download the Super-VAE checkpoints
```bash
hf download MitakaKuma/SMILE --include="autoencoder/*" --local-dir "./ckpt" 
```

## Download the Classification model:
```bash
hf download MitakaKuma/SMILE --include="classifier/*" --local-dir="./ckpt"
```

## Download the Segmentation model:
```bash
hf download MitakaKuma/SMILE --include="segmenter/*" --local-dir="./ckpt"
```

## Download the SMILE checkpoints
```bash
hf download MitakaKuma/SMILE --include="SMILE/*" --local-dir="./ckpt"
```
