# Model weights

The trained weights (`.pth`) are not stored in the repository because of their size. They can be attached to a GitHub release or reproduced with the training code in `training/`.

| File | Architecture | Training set |
|---|---|---|
| `simple_cnn_set2.pth` | SimpleCNN | Set 2 |
| `simple_cnn_set3.pth` | SimpleCNN | Set 3 |
| `mobilenet_v3_set2.pth` | MobileNet-V3-Small (ImageNet pre-trained, fine-tuned) | Set 2 |
| `mobilenet_v3_set3.pth` | MobileNet-V3-Small (ImageNet pre-trained, fine-tuned) | Set 3 |
| `squeezenet_set2.pth` | SqueezeNet 1.1 (ImageNet pre-trained, fine-tuned) | Set 2 |
| `squeezenet_set3.pth` | SqueezeNet 1.1 (ImageNet pre-trained, fine-tuned) | Set 3 |

All models take 224 × 224 RGB input normalised with the ImageNet mean and standard deviation and output 4 classes in the order `Down, Left, Right, Up`.
