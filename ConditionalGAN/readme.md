## 1. Introduction

<p align="center"><img src="https://github.com/user-attachments/assets/dd288c31-8928-4bc9-a4a0-91d71a698c4c" width="60%" height="60%"></p>

Conditional GAN은 Generator와 Discriminator를 학습하기 위하여 데이터셋에 있는 레이블을 사용합니다. 이러한 특성 덕분에 Conditional GAN의 Generator는 레이블 정보를 주입하면 원하는 가짜 이미지를 생성할 수 있습니다.
Generator는 노이즈 벡터 $z$와 레이블 $y$를 입력으로 받아 가짜 이미지 $G(z|y)$을 생성합니다. Discriminator의 경우 진짜 이미지와 레이블 또는 가짜 이미지와 레이블을 입력으로 받아 $D(x|y)$를 생성합니다. 레이블 정보를
추가적으로 입력하는 것을 제외하고 나머지 학습 방법은 Naive GAN과 동일합니다.

## 2. Train

데이터셋은 mnist 데이터셋을 활용합니다.

- 학습을 위해 아래와 같은 명령어로 train.py를 실행시켜주세요. args에 대한 자세한 내용은 코드를 참고해주세요.
```python
python train.py --[args]
```

## 3. Inference
학습이 완료되면 inference.py를 이용하여 추론을 수행할 수 있습니다. args에 대한 자세한 내용은 inference.py를 참고해주세요.
```python
python inference.py --[args]
```

## 4. MNIST Result

<p align="center"><img src="https://github.com/user-attachments/assets/cdd12310-7798-4ec6-b5ee-e6e44bb3a702" width="90%" height="90%"></p>

위의 결과 이미지는 Conditional GAN의 Generator를 사용하여 클래스 정보를 조건으로하여 9개의 숫자를 생성한 결과 입니다. mode collapse 현상 없이 10개의 숫자가 모두 잘 생성된 것을 확인할 수 있습니다.
