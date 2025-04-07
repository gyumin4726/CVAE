# 과제

## 과제 목표
뼈대코드에 자신만의 코드조각을 추가하여 Conditional Variational Autoencoder (이하, CVAE)를 구현한다.

뼈대코드는 `cvae_skeleton.zip` 파일로 제공되며, 뼈대코드에 자신만의 코드조각를 추가하여 아래 사항을 수행한다. 
- CVAE 모델 설계
- CVAE 모델 학습
- CVAE 생성 네트웍 활용
- 보고서 작성

### 과제 제출물
다음을 포함하는 `학번.zip` 파일을 가상대학에 제출한다.
- 수정한 소스코드 (`*.py`)
- 보고서 파일 (`학번.pdf`)
  
#### 주의사항
- 수정한 소스코드를 제출할 때에는 반드시, 모델의 학습과 구동에 직접적으로 필요하지 않은 폴더 및 파일은 모두 삭제한 후, `.zip` 파일로 묶어서 제출하도록 한다.
  -  `__pycache__`, `MNIST_DATASET`, `lightning_logs`, `cvae.ckpt` 등
-  위 사항을 지키지 않았을 경우, 감점 대상이 됨.
 
### 과제 수행 환경 설치
[과제 수행 환경 설치](./INSTALL/INSTALL.md)를 참고하여 각종 라이브러리를 설치한다.

## 과제 내용
### CVAE 모델 설계
`model.py` 파일을 수정하여, MNIST 데이터셋에서 class label에 해당하는 0-9 사이의 digit을 조건으로 활용하여 훈련하고 이를 생성 모델로 활용할 수 있는 CVAE 모델을 설계한다.

선형 레이어들로 구성된 multiplayer perceptron (이하, MLP)로 구성하는 것으로 기본으로 하며, 자신의 기호에 따라 convolutional neural network (이하, CNN)으로 구성해도 좋다.

### CVAE 모델 학습
`train.py` 파일을 다음과 같이 구동시켜, 자신이 설계한 CVAE 모델이 학습되도록 한다. 

```shell
python train.py
```
학습 후, CVAE 모델의 파라미터는 `cvae.ckpt`로 저장한다.

### CVAE 생성 네트웍 활용
`gradio_test.py` 파일을 다음과 같이 구동시켜, 자신이 설계한 CVAE 모델의 `decoder`가 생성 네트웍으로 작동되는지 확인한다.

```shell
python gradio_test.py
```

`gradio_test.py` 파일은 `cvae.ckpt` 파일을 읽어들여, CVAE 모델의 파라미터를 훈련이 끝난 직후 상태로 만든 다음, CVAE 생성 네트웍을 웹브라우져에서 구동되는 Gradio UI를 통해 활용할 수 있도록 설계되어 있다.


`python gradio_test.py` 명령을 수행하고 명령어 창을 관찰하면 다음과 같은 메시지가 뜬다. 
```shell
* Running on local URL:  http://127.0.0.1:7860
```
이 때, 해당 URL을 클릭하거나 웹브라우져의 주소창에 해당 URL을 입력하게 되면, Gradio UI가 뜨면서 CVAE 생성 네트웍이 의미있게 돌아가는지 테스트 할 수 있다. 

`gradio_test.py` 파일은 거의 완성된 형태이니 조금씩 자신의 과제에 맞게 수정해서 사용하면 된다.


### 보고서 작성
CVAE 모델 설계, 훈련, 결과 분석을 담은 보고서를 `학번.pdf`파일로 작성한다.

