## 1. Introduction
Unsupervised Image-to-Image Translation Networks(UNIT)는 unpaired한 데이터들간 서로 공유하고 있는 중요 특징(shared latent space)을 이용해 style을 변환시켜주는 연구이며 2017년 NVIDIA에서 발표한 논문입니다.

### Shared Latent Space

<p align="center"><img src="https://github.com/user-attachments/assets/ad38cfde-97be-4da5-bded-9eae4408d5b0" width="40%" height="40%"></p>

Shared Latent Space는 UNIT의 가장 핵심 개념으로 두 domain간에는 서로 공유하고 있는 latent space가 있다고 가정합니다.
$X_1$의 domain에서 $X_2$의 domain 사이에서 공유하고 있는 latent space $z$가 있다고 가정하면 위의 그림처럼 $x_1$의 이미지에서 Encoder $E_1$을 통해 $z$가 되고 이 $z$를 가지고 Generator $G_2$를 통해
$x_2$ 이미지를 생성할 수 있습니다. 반대의 경우도 마찬가지 입니다.

### Weight Sharing

<p align="center"><img src="https://github.com/user-attachments/assets/86c36fce-e81c-4e5b-938b-b903f10161a5" width="60%" height="60%"></p>

아키텍처는 VAE와 GAN을 혼합한 형태로 중간에 latent spoace $z$를 공유한다는 것이 UNIT의 key idea 입니다.
$x_1$과 $x_2$는 각각 $E_1$과 $E_2$를 통해 shared latent space $z$로 인코딩되며 이 $z$를 가지고 $G_1$, $G_2$를 통해 $x'_1$과 $x'_2$ 이미지를 생성합니다.
$E_1$, $E_2$의 마지막 부분의 Convolution layer는 sharing하게 설계하여 shared $z$를 추출하고 $G_1$과 $G_2$의 처음 부분 몇 개의 Convolution layer도 sharing하게 설계합니다.
Generator와 Discriminator는 Naive GAN과 같은 형태입니다.

### Training Loss

<p align="center"><img src="https://github.com/user-attachments/assets/c321bd74-b31b-4739-9601-ebeded85580e" width="60%" height="60%"></p>

VAE와 GAN을 혼합으로 사용했기 때문에 Loss는 위와 같이 VAE loss와 GAN loss를 합쳐놓은 형태입니다. 그리고 CycleGAN에서 사용한 Cycle-Consistency loss를 사용하며 이는 $x_1$ 이미지를 이용하여 $G_2$를 통해 $x'_2$ 이미지를 생성하고
$G_1$를 통해 $x'_1$ 이미지를 재생성하면 이는 $x_1$과 같아야한다는 것입니다.
GAN 학습 목표와 동일하게 $E_1,G_1,E_2,G_2$를 freeze하여 $D_1, D_2$ 먼저 학습하고, $D_1, D_2$를 freeze하고 Encoder, Generator를 학습하는 구조입니다.

## 2. Dataset Preparation
데이터셋은 apple2orange, facade, summer2winter_yosemite을 사용하여 학습을 진행하였습니다.
다운로드 링크는 아래와 같습니다.
- apple2orange : https://www.kaggle.com/datasets/balraj98/apple2orange-dataset
- facade : https://www.kaggle.com/datasets/balraj98/facades-dataset
- summer2winter : https://www.kaggle.com/datasets/balraj98/summer2winter-yosemite

데이터셋의 폴더 구조는 아래와 같습니다.
```python
data
├── apple2orange
│   ├── testA
│   ├── testB
│   ├── trainA
│   └── trainB
```

## 3. Train
- 학습을 위해 아래와 같은 명령어로 train.py를 실행시켜주세요. args에 대한 자세한 내용은 코드를 참고해주세요.
```python
python train.py --[args]
```

## 4. Inference
학습이 완료되면 inference.py를 참고하여 학습 완료된 모델의 가중치를 로드하여 추론을 수행할 수 있습니다.

## 5. Result

### Facades
<p align="center"><img src="https://github.com/user-attachments/assets/38b53ad3-e081-4607-8351-912e0194410b" width="40%" height="40%"></p>

### Facades
<p align="center"><img src="https://github.com/user-attachments/assets/9e47c865-a7ca-4e3f-86ac-fbe4e9f5b239" width="40%" height="40%"></p>

### Summer2Winter
<p align="center"><img src="https://github.com/user-attachments/assets/8b16569d-00ab-44cb-9516-e6e8e144f66f" width="40%" height="40%"></p>
