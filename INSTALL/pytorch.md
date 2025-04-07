# 파이토치 설치

파이토치 설치 방법을 정리한 문서이며, 다음 라이브러리 버젼을 기준으로 작성되었다.
- (옵션) CUDA 12.6.3
- (옵션) cuDNN 8.9.7.29 (for CUDA 12.6)
- Miniforge3
- 파이토치 2.6.0

## (옵션) CUDA 설치
NVIDIA 그래픽카드가 있다면, [CUDA 삭제 및 설치](https://stat-thon.tistory.com/104)을 참고해서 CUDA 및 cuDNN 라이브러리를 설치한다. 

NVIDIA 그래픽카드가 없거나, 설치가 너무 복잡하다고 생각되면 이 부분을 건너 띄워도 좋다.


## 파이토치 설치

[Miniforge 설치하기](miniforge.md) 문서를 참고하여 Miniforge 기반의 conda 가상환경을 설치한다.

## 파이토치 위한 conda 가상환경 구축

다음과 같이 `pytorch`라는 이름의 conda 가상환경을 `python=3.10` 버젼 기준으로 만들도록 한다.
```sh
conda create -n pytorch python=3.10
```

아래와 같이 conda 가상환경 `pytorch`를 활성화 하도록 하자.
```sh
conda activate pythorch
```

`pytorch` 가상환경에는 지금 해당 가상환경을 만들 당시에 깔린 python 3.10 버전만 있다. 이제, `pytorch` 가상환경에 최신 PyTorch 라이브러리를 다음과 같이 설치하자. CUDA 설치 유무에 따라 적절한 방법을 선택하면 된다.


###### PyTorch (CUDA 지원) 설치

[PyTorch 홈페이지](https://pytorch.org/)를 참고하여 자신의 OS 환경에 맞는 방식으로 파이토치를 설치하도록 하자.

Windows PC에 NVIDIA 그래픽카드가 있고 CUDA 및 cuDNN이 설치되어 있다면, 다음과 같이 CUDA 가속이 지원되는 PyTorch를 설치하도록 한다.
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

그렇지 않은 경우, 다음과 같이 CPU 지원 PyTorch를 설치하도록 한다.
```sh
pip3 install torch torchvision torchaudio
```


###### PyTorch가 CUDA 인식 여부 확인

python을 구동시킨 후, 다음 코드를 실행해 보도록 하자.
```python
import torch

print(torch.__version__)
print("GPU is", "available" if torch.cuda.is_available() else "not available")
```

CPU만 지원하는 PyTorch를 설치했을 경우, 다음과 같은 메시지가 뜬다.
```
2.6.0+cpu
GPU is not available
```

GPU를 지원하는 PyTorch를 설치했을 경우, 다음과 같은 메시지가 뜬다.
```
2.6.0
GPU is available
```
