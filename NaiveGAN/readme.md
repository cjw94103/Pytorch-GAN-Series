## 1. Introduction

<p align="center"><img src="https://github.com/user-attachments/assets/b64abe82-7a05-466d-a85e-8b4a98306ab4" width="60%" height="60%"></p>

Naive GAN은 2015년 이안 굿펠로우 외 7인이 이미지 생성을 위한 새로운 패러다임을 제시합니다. GAN은 Generator와 Discriminator라는 두 개의 Deep neural network가 Adversarial Training을 통해 학습을 수행합니다.
먼저 Generator은 학습 데이터셋에 있는 실제 데이터와 구분이 되지 않을 정도의 가짜 데이터를 생성하는 역할을 합니다. Discriminator는 Generator가 만든 가짜 데이터를 학습 데이터셋에 있는 실제 데이터와 구별하는 역할을 수행합니다.
즉, Generator는 Discriminator가 실제, 가짜 이미지를 구별하지 못할 정도의 완벽한 이미지를 생성하는 것을 목표로 학습하며, Discriminator는 실제, 가짜 이미지를 완벽하게 구별하기 위한 목표로 학습합니다.
GAN은 두 네트워크가 서로 적대적인 입장에서 Nash Equilibrium 상태에 도달하는 것을 목표로 합니다.


<p align="center"><img src="https://github.com/user-attachments/assets/581db00a-7c7b-4dc9-a2c7-9a8bbe2be090" width="60%" height="60%"></p>

위의 수식은 GAN의 Loss function입니다.

- 위의 수식에서 왼쪽 Term은 Discriminator가 실제 데이터에서 샘플링한 데이터를 입력으로 받으면, 1에 가까운 값을 왼쪽 Term은 0에 가까워지고 최소화가 됩니다.
- 오른쪽 Term은 Generator가 노이즈 벡터 $z$를 입력으로 받아 생성한 가짜 이미지를 Discriminator에 입력으로 사용하여 실제, 가짜 이미지에 대한 이진 분류를 수행합니다. 즉, $D(G(z))$의 값이 1일때, 즉, Discriminator가 가짜 이미지를 진짜라고 분류했을 경우 최대화가 됩니다.
- 이러한 Adversarial Training을 통해 (최소화 vs 최대화) Nash Equilibrium 상태에 도달하는 것이 GAN 학습의 최종 목표가 됩니다.

## 2. Train
데이터셋은 mnist 데이터셋을 활용합니다. 

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

<p align="center"><img src="https://github.com/user-attachments/assets/ebce3c81-4716-4b62-99cb-a629b9ddc842" width="60%" height="60%"></p>

위의 결과 그림을 보면 Generator는 mode collapse 현상이 발생합니다. 보다 좋은 방법인 DCGAN을 사용하면 이러한 mode collapse를 줄일 수 있습니다.
