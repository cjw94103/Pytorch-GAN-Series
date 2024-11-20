# 1. Introduction
<p align="center"><img src="https://github.com/user-attachments/assets/2d4ba483-f60e-4734-9bea-591b09c8a3be" width="90%" height="90%"></p>

Multimodal Unsupervised Image-to-Image Translation는 UNIT 논문을 발전한 형태로 unimodal이 아닌 multimodal로의 변환이 가능하도록 구현한 논문입니다. UNIT와 마찬가지로 Unsupervised Image-to-Image translation이며
하나의 이미지는 content와 style을 가지고 있어 content는 유지한채 style만 변환해줌으로서 다양한 style로 자유로운 변환이 가능함을 보여준 논문입니다. 
NVIDIA에서 2018년에 발표한 논문으로, content와 style의 개념을 처음 소개함으로 이후에 나오는 Image-to-Image Translation 논문들의 baseline에 해당하는 논문입니다. 

## Loss function
<p align="center"><img src="https://github.com/user-attachments/assets/2cacfd0a-97a2-4240-8527-eddebbb85ad8" width="90%" height="90%"></p>

### Bidirectional reconstruction loss
image의 reconstruction은 image -> latent space -> image의 형태로 수행합니다. 이미지 $x_1$이 있을 때 Encoder를 통해 content $c_1$과 style $s_1$ 코드가 나오고 이 두 코드를 Decoder에 넣으면 $x_1'$이 생성되며
이 output은 $x_1$과 $x_1'$이 동일해야 합니다.

<p align="center"><img src="https://github.com/user-attachments/assets/aa46b97e-69ae-4a7d-bb41-17bf538e842e" width="60%" height="60%"></p>

### Latent reconstruction
latent의 reconstruction은 latent space -> image -> latent space의 형태로 수행합니다. 이미지의 reconstruction 뿐만 아니라 latent의 reconstruction도 동일해야 합니다.
위의 그림에서 (b)처럼 두 domain 간에 이루어지는 과정으로 먼저 $x_1$ 이미지의 content $c_1$과 $x_2$ 이미지의 style $s_2$를 결합하여 $x_2'$ 이미지를 생성합니다.
생성된 $x_2'$ 이미지를 다시 Encoder에 입력하면 content code $c_1'$과 style code $s_2'$이 생성되고, 여기서 $c_1'$은 이미지 $x_1$의 content인 $c_1$과 같아야하고, $s_2'$은 target image인 $x_2$의 style인 $s_2$와 같아야 합니다.
이를 수식으로 표현하면 아래와 같습니다.

<p align="center"><img src="https://github.com/user-attachments/assets/e45d1b35-0851-48e4-85f2-16685289429f" width="60%" height="60%"></p>

### Adversarial Loss
기본적으로 GAN을 사용하기 때문에 Adversarial loss를 사용해야 하며 $x_1$ 이미지의 content code $c_1$과 $x_2$ 이미지의 style code $s_2$를 가지고 Decoder를 통해 $x_2'$ 이미지를 생성합니다.
$x_2'$은 Discriminator를 속여야하므로 이를 수식으로 표현하면 아래와 같습니다.

<p align="center"><img src="https://github.com/user-attachments/assets/f64d4eed-f130-4999-9e59-d56de7e6defa" width="80%" height="80%"></p>

### Loss Function
MUNIT의 최종 loss formulation은 아래와 같습니다. Bidirectional reconstruction loss, Latent reconstruction, Adversarial Loss의 weighted sum으로 구성됩니다. $\lambda_x, \lambda_c, \lambda_s$ 는 사용자가 정해주어야하는
하이퍼파라미터입니다. 논문에서는 각각 10.0, 1.0, 1.0을 사용하였습니다.

<p align="center"><img src="https://github.com/user-attachments/assets/779b410f-9f35-4be8-8834-bbc868cb3c69" width="80%" height="80%"></p>

# 2. Dataset Preparation
데이터셋은 edges2shoes, summer2winter_yosemite을 사용하였습니다. 다운로드 링크는 아래와 같습니다.

- edges2shoes : https://www.kaggle.com/datasets/balraj98/edges2shoes-dataset
- summer2winter : https://www.kaggle.com/datasets/balraj98/summer2winter-yosemite

데이터셋의 폴더 구조는 아래와 같습니다.

```python
data
├── edges2shoes
│   ├── testA
│   ├── testB
│   ├── trainA
│   └── trainB
```

# 3. Train
- 학습을 위해 아래와 같은 명령어로 train.py를 실행시켜주세요. args에 대한 자세한 내용은 코드를 참고해주세요.
```python
python train.py --[args]
```

# 4. Inference
학습이 완료되면 inference.py를 참고하여 학습 완료된 모델의 가중치를 로드하여 추론을 수행할 수 있습니다.

# 5. Result

## edges2shoes

<p align="center"><img src="https://github.com/user-attachments/assets/7c4f1f1e-a502-47ef-b75e-a089b95b9218" width="60%" height="60%"></p>

## summer2winter

<p align="center"><img src="https://github.com/user-attachments/assets/0d05f3c5-5dfc-4832-8630-ddbc066f72a7" width="60%" height="60%"></p>
