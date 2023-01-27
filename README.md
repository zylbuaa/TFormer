# TFormer
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)   

## Introduction
The official implementation of "TFormer: A throughout fusion transformer for multi-modal skin lesion diagnosis"

![Our Network Structure](graphical_abstract.png)


## Enviroments
- Windows/Linux both support
- python 3.8
- PyTorch 1.9.0
- torchvision

## Prepare dataset
Please at first download datasets [Derm7pt](https://derm.cs.sfu.ca/Download.html) and then download the pretrained model of swin-tiny on ImageNet-1k from [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth). Save the model into the folder "./models/swin_transformer".

## Run details
To train our `TFormer`, run:
```
python train.py --dir_release "your dataset path" --epochs 100 --batch_size 32 --learning_rate 1e-4 --cuda True
```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details

## Acknowledgement
Our code borrows a lot from:
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [Derm7pt](https://github.com/jeremykawahara/derm7pt)

