## 1. Introduction

<p align="center"><img src="https://github.com/user-attachments/assets/f123617e-4253-4ced-ba47-e9d4e5e6c6c0" width="100%" height="100%"></p>

DCGAN은 더 좋고 효율적인 이미지 생성을 위해 Naive GAN을 개선한 방법입니다. DCGAN은 안정적인 이미지 생성을 위해 NaiveGAN에서 아래와 같은 네 가지 사항을 개선하였습니다.

- MaxPooling 연산을 사용하는 대신 Convolution의 Stride를 통해 생성되는 이미지의 크기 조절
- Dense Layer 대신 Generator에서 노이즈 벡터 $z$를 넣는 첫 번째 layer와 Discriminator에서 출력 결과를 판단하는 마지막 Softmax 연산을 제외하고 Dense Layer를 전부 제외함
- Internal Covariance Shift 현상을 방지하기 위해 BatchNormalization layer 추가
- ReLU 대신 LeakyReLU 사용

이러한 개선 사항으로 DCGAN은 GAN에 비해 mode collapse 현상이 많이 발생하지 않고 더 좋은 품질의 이미지를 생성할 수 있습니다.

## 2. Train
학습 데이터로 MNIST 데이터셋을 사용합니다.

- 학습을 위해 아래와 같은 명령어로 train.py를 실행시켜주세요. args에 대한 자세한 내용은 코드를 참고해주세요.
```python
python train.py --[args]
```
## 3. Inference
학습이 완료되면 inference.py를 이용하여 추론을 수행할 수 있습니다. args에 대한 자세한 내용은 inference.py를 참고해주세요.
```python
$ python inference.py --[args]
```

## 4. MNIST Result
<p align="center"><img src="https://github.com/user-attachments/assets/c38ae6c0-be33-43e8-9ddb-1b9a66de1486" width="60%" height="60%"></p>

NaiveGAN에 비해 mode collapse 현상이 발생하지 않으며, pepper and salt 노이즈 없이 깨끗한 이미지가 생성됩니다.
