## 1. Introduction

<p align="center"><img src="https://github.com/user-attachments/assets/ad299854-aa0b-4bc8-af10-cfc6fa889eb2" width="80%" height="80%"></p>

Naive GAN은 random noise vector $z$에서 output image $y$에 대하여 $G : z \to y$로의 mapping을 학습하는 generative model 입니다.
대조적으로, Conditional GAN은 실제 이미지 $x$와 random noise vector $z$에서 $\{x, z\} \to y$로의 mapping을 학습합니다. 즉, Conditional GAN은 $x$를 조건으로 하여 (예를 들어, 레이블) random noise vector $z$의 출력을 제어할 수 있습니다.
Pix2Pix는 Conditional GAN의 변형으로써 $x$를 조건으로하여 random noise vector $z$로부터 다른 domain의 이미지를 생성가능하게 합니다. 즉, Image-to-Image Translation을 가능하게 하는 기법입니다.


<p align="center"><img src="https://github.com/user-attachments/assets/4a8c0b46-dd5f-43b8-b735-170eb9fc0bb6" width="50%" height="50%"></p>

위의 식은 Pix2Pix의 Objective입니다. $\mathcal L_{cGAN}$은 Conditional GAN의 Objective와 동일하며 $\mathcal L_{L1}$의 식은 아래와 같습니다.

<p align="center"><img src="https://github.com/user-attachments/assets/2f160fc7-e57c-42ca-947b-28b4d7127f79" width="40%" height="40%"></p>

즉, $x$를 조건으로하여 random noise vector $z$로부터 생성된 이미지 $G(x, z)$와 $y$에 대한 L1 loss를 최소화하여 생성된 이미지가 실제 이미지와 최대한 같아지도록 강제합니다. $\lambda$는 L1 loss에 대한 가중치로서
이 저장소의 실험은 모두 100으로 설정하여 진행합니다.

## 2. Dataset Preparation
데이터셋은 https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset 에서 배포한 데이터셋을 사용합니다. 데이터셋의 구성은 아래와 같습니다.

- Facades
- Cityscapes
- Maps
- Edges to shoes

데이터셋을 다운로드 받아 ./data 폴더에 넣어주세요.

## 3. Train
- 학습을 위해 아래와 같은 명령어로 train.py를 실행시켜주세요. args에 대한 자세한 내용은 코드를 참고해주세요.
```python
python train.py --[args]
```

## 4. Inference
학습이 완료되면 inference.py를 이용하여 추론을 수행할 수 있습니다. args에 대한 자세한 내용은 inference.py를 참고해주세요.
```python
$ python inference.py --[args]
```

## 5. Result

### Facades
<p align="center"><img src="https://github.com/user-attachments/assets/cff4f7a0-5b2d-4d6f-aa6e-4c7f7bcd2a3b" width="40%" height="40%"></p>

### Cityscapes
<p align="center"><img src="https://github.com/user-attachments/assets/3f18d565-62d4-4b16-a866-1bd43b641918" width="40%" height="40%"></p>

### Maps
<p align="center"><img src="https://github.com/user-attachments/assets/affd2bff-61d4-4bf2-9672-101e8b1d0f93" width="40%" height="40%"></p>

### Edges to shoes
<p align="center"><img src="https://github.com/user-attachments/assets/9be53f82-998a-4a30-ba97-8e024df2ca26" width="40%" height="40%"></p>
