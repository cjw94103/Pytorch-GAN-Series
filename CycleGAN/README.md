# 1. Introduction
![cyclegan figure](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/b81f14c3-605c-47a4-8848-f5f67976fb85)
\
CycleGAN은 image-to-image translation의 대표적 모델로 paired example 없이 $X$라는 domain으로부터 얻은 이미지를 target domain $Y$로 translation하는 방법입니다.
CycleGAN의 목표는 Adversarial Loss를 통해, $G(x)$로부터의 이미지 데이터의 분포와 $Y$로부터의 이미지 데이터의 분포를 구별할 수 없도록 forward mapping $G:X \to Y$을 학습하고 constraint를 위해 inverse mapping $F:Y \to X$를 학습합니다.
image translation을 위하여 inverse mapping $F(G(x))$가 $x$와 같아지도록 Cycle Consistency Loss를 사용합니다.
추가적으로 이미지 $x$와 생성된 이미지 $x'$, 이미지 $y$와 생성된 이미지 $y'$이 같아지도록 강제하는 Identity Loss를 사용합니다. \
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
먼저 config.json 파일을 만들어야 합니다. make_config.ipynb 파일을 참고하여 config 파일을 만들어주세요.
학습 또는 추론에 사용 할 특정 GPU의 선택을 원하지 않는 경우 코드에서 os.environ["CUDA_VISIBLE_DEVICES"]="1"를 주석처리 해주세요.

- Single GPU인 경우 config에서 multi_gpu_flag로 설정하고 아래와 같이 명령어를 실행시켜주세요.
```python
python train.py --config_path /path/your/config_path
```
- Multi GPU의 경우 Pytorch에서 지원하는 DistributedDataParallel로 분산 학습을 구현하였습니다. multi_gpu_flag를 True, Port_num를 입력하고 아래와 같이 명령어를 실행시켜주세요.
```python
python train_dist.py --config_path /path/your/config_path
```
# 4. Inference
학습이 완료되면 inference.ipynb를 참고하여 학습 완료된 모델의 가중치를 로드하여 추론을 수행할 수 있습니다.
# 5. 학습 결과
각 테스트 데이터셋에 대한 translation 결과를 보여줍니다.
## Apple2Orange
![apple2orange](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/022fb04a-bbef-4ff7-9f86-973cf70a596c)
## Facade
![facade](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/89bf38b9-c9ef-4b36-95b7-626fc181d7b6)
## Horse2Zebra
![horse2zebra](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/d43f925b-9a9b-449b-b533-48ba22759e3b)
## Monet2Photo
![monet2photo](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/636056ad-03b5-41c8-bfc6-fedc7bd3ba83)
## Summer2Winter
![summer2winter](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/60936d84-434d-44c3-bb18-c95ae8229d25)
