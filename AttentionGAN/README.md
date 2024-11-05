# 1. Introduction
![att](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/68764dd4-e4e8-4925-b054-277a307066a8)
\
\
AttentionGAN은 image-to-image translation 모델로 paired example 없이 $X$라는 domain으로부터 얻은 이미지를 target domain $Y$로 translation하는 방법입니다.
AttentionGAN의 목표는 Adversarial Loss를 통해, $G(x)$로부터의 이미지 데이터의 분포와 $Y$로부터의 이미지 데이터의 분포를 구별할 수 없도록 forward mapping $G:X \to Y$을 학습하고 constraint를 위해 inverse mapping $F:Y \to X$를 학습합니다.
image translation을 위하여 inverse mapping $F(G(x))$가 $x$와 같아지도록 Cycle Consistency Loss를 사용합니다.
추가적으로 이미지 $x$와 생성된 이미지 $x'$, 이미지 $y$와 생성된 이미지 $y'$이 같아지도록 강제하는 Identity Loss를 사용합니다.
또한 Generator 아키텍처에 foreground, background attention을 추가하여 target object의 디테일한 translation을 가능하게 합니다.\
\
구현은 https://github.com/eriklindernoren/PyTorch-GAN 을 주로 참고하였으며 single, multi-gpu에서 구동될 수 있도록 Pytorch 2.1.0버전으로 구현되었습니다.
# 2. Dataset Preparation
데이터셋은 apple2orange, facade, horse2zebra, monet2photo, summer2winter_yosemite을 사용하여 학습을 진행하였습니다.
다운로드 링크는 아래와 같습니다.
- apple2orange : https://www.kaggle.com/datasets/balraj98/apple2orange-dataset
- facade : https://www.kaggle.com/datasets/balraj98/facades-dataset
- horse2zebra : https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset
- monet2photo : https://www.kaggle.com/datasets/balraj98/monet2photo
- https://www.kaggle.com/datasets/balraj98/summer2winter-yosemite

\
데이터셋의 폴더 구조는 아래와 같습니다.
```python
data
├── apple2orange
│   ├── testA
│   ├── testB
│   ├── trainA
│   └── trainB
```
# 3. Train
- Single GPU인 경우 train.py를 실행하세요. args에 대한 자세한 내용은 코드를 참고해주세요
```python
python train.py --[args]
```
- Multi GPU의 경우 Pytorch에서 지원하는 DistributedDataParallel로 분산 학습을 구현하였습니다. multi_gpu_flag를 True, Port_num를 입력하고 train_dist.py를 실행하세요.
```python
python train_dist.py --[args]
```
# 4. Inference
학습이 완료되면 inference.ipynb를 참고하여 학습 완료된 모델의 가중치를 로드하여 추론을 수행할 수 있습니다.
# 5. 학습 결과
각 테스트 데이터셋에 대한 translation 결과를 보여줍니다.
## Apple2Orange
![apple2orange](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/a0ec14cd-b973-41fa-81db-8229bcce8a11)
## Facade
![facade](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/ae0e3061-2ef0-40a6-85b5-12ea07ba3e53)
## Horse2Zebra
![horse2zebra](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/99cd6b6a-7b69-4158-a4f3-1a5a26a0b654)
## Monet2Photo
![monet2photo](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/d3a3e115-f180-4e3b-8037-96c00af97f42)
## Summer2Winter
![summer2winter](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/cd5f982e-aece-46da-a54c-507ed3eaff53)
