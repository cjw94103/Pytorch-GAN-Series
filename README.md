<p align="center"><img src="https://github.com/user-attachments/assets/772ea16c-ca0b-4ac4-a71a-c91fcf331cd5" width="35%" height="35%"></p>

## 1. Introduction
GAN (Generative Adversarial Network)의 Pytorch 구현 모음입니다. 모델 아키텍처는 논문의 내용대로 완전히 똑같이 구현되지는 않지만 학습 루프, 하이퍼파라미터 등 핵심 아이디어를 다루는데 집중합니다. 
구현할 GAN의 종류는 아래와 같습니다.

- Noise to Image Translation
- 2-Domain I2I Translation
- Multi-Domain I2I Translation

code의 각 GAN 모델에 대한 폴더를 들어가시면 학습 및 추론 방법이 작성되어 있습니다.

## 2. Table of Contents
- Noise to Image Translation
  - NaiveGAN
  - DCGAN
  - ConditionalGAN
- 2-Domain I2I Translation
  - Pix2Pix
  - CycleGAN
  - AttentionGAN
- Multi-Domain I2I Translation
