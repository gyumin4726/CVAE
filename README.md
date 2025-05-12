# Conditional VAE (CVAE)

MNIST 데이터셋을 사용한 조건부 이미지 생성을 위한 Conditional VAE 구현 프로젝트입니다.

## VAE의 발전 과정

### VAE (Variational AutoEncoder)
- 생성 모델의 기본 구조
- 인코더와 디코더의 구조를 가진 오토인코더
- 잠재 공간에서의 확률적 샘플링을 통한 이미지 생성
- 인코더: 입력 이미지를 잠재 공간의 분포로 변환
- 디코더: 잠재 벡터를 원본 이미지로 복원

### CVAE (Conditional VAE)
- VAE에 조건부 생성 기능을 추가한 모델
- 클래스 레이블을 조건으로 특정 클래스의 이미지 생성 가능
- 인코더와 디코더 모두에 클래스 정보 주입
- MNIST 데이터셋의 경우 0-9까지의 숫자를 조건부로 생성
- PyTorch Lightning의 LightningModule을 상속받아 구현

## 프로젝트 구조

```
.
├── model.py          # CVAE 모델 구현
├── dataset.py        # 데이터셋 로드 및 전처리
├── train.py          # 모델 학습 스크립트
├── test.py           # 모델 테스트 스크립트
├── gradio_test.py    # Gradio 웹 인터페이스
└── requirements.txt  # 프로젝트 의존성
```

## 설치 방법

1. 저장소 클론
```bash
git clone [https://github.com/gyumin4726/CVAE]
cd CVAE
```

2. 가상환경 생성 및 활성화
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

### 모델 학습
```bash
python train.py
```

### 모델 테스트
```bash
python test.py
```

### 웹 인터페이스 실행
```bash
python gradio_test.py
```

## 주요 기능

- MNIST 데이터셋을 사용한 조건부 이미지 생성
- PyTorch Lightning 기반 모델 구현
- Gradio를 통한 웹 기반 인터페이스 제공
- 사용자 정의 잠재 벡터를 통한 이미지 생성 제어

## 기술 스택

- PyTorch
- PyTorch Lightning
- Gradio
- MNIST Dataset

## 모델 아키텍처

### 인코더
1. **입력 처리**
   - 이미지: 784차원 (28×28)
   - 클래스 레이블: 10개 클래스 (one-hot encoding)
   - 결합: 794차원 벡터 (784 + 10)

2. **레이어별 구조**
   ```
   입력: 794차원
   ↓
   Layer 1: 794 → 512
   - Linear + LayerNorm + ReLU + Dropout(0.25)
   ↓
   Layer 2: 512 → 256
   - Linear + LayerNorm + ReLU + Dropout(0.25)
   ↓
   Layer 3: 256 → 128
   - Linear + LayerNorm + ReLU
   ↓
   출력: 
   - mean: 128 → 2 (Linear)
   - logvar: 128 → 2 (Linear)
   ```

### 디코더
1. **입력 처리**
   - 잠재 벡터(z): 2차원
   - 클래스 레이블: 10개 클래스 (one-hot encoding)
   - 결합: 12차원 벡터 (2 + 10)

2. **레이어별 구조**
   ```
   입력: 12차원
   ↓
   Layer 1: 12 → 128
   - Linear + LayerNorm + ReLU + Dropout(0.25)
   ↓
   Layer 2: 128 → 256
   - Linear + LayerNorm + ReLU + Dropout(0.25)
   ↓
   Layer 3: 256 → 512
   - Linear + LayerNorm + ReLU + Dropout(0.25)
   ↓
   Layer 4: 512 → 784
   - Linear + Sigmoid
   ↓
   출력: 784차원 (28×28)
   ```

## 하이퍼파라미터

- 잠재 벡터 차원: 2
  - 인코더의 출력 차원으로 고정
  - 이미지 생성의 다양성을 제어하는 파라미터
- 배치 크기: 100
- 학습률: 0.001
- Adam 옵티마이저
- 에포크 수: 10
- 손실 함수: MSE Loss

## 주의사항

- 학습 시간은 하드웨어 사양에 따라 달라질 수 있습니다.
- 웹 인터페이스는 기본적으로 localhost:7860에서 실행됩니다.
- 메모리 사용량을 고려하여 `dataset.py`의 `batch_size`를 조정할 수 있습니다.

## 문제 해결

### 일반적인 문제
1. 메모리 부족 오류
   - `dataset.py`에서 `batch_size` 값을 줄임
2. Gradio 실행 오류
   - 포트가 이미 사용 중인 경우 `gradio_test.py`에서 포트 번호 변경
   ```python
   # gradio_test.py 마지막 줄을 다음과 같이 수정
   demo.launch(server_port=7861)  # 7860 대신 다른 포트 번호 사용
   ```
