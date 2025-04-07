# 과제 수행 환경 설치

## Conda 가상환경 구축
Conda 가상환경에서 많이 쓰이는 Miniforge를 아래 문서를 참고하여 설치하자.

## PyTorch, Lightning, Gradio 라이브러리 설치

Conda 가상환경에서 `pytorch`를 활성화 한 후, 아래 문서를 참고하여 과제에 필요한 라이브러리들을 설치하자.
- [PyTorch 설치](./pytorch.md)
- [Lightning 설치](./lightning.md)
- [Gradio 설치](./gradio.md)


python을 구동시킨 후, 다음 코드를 실행했을 때, 에러 없이 버젼 정보가 출력되면 제대로 설치된 것이다.

```python
import torch
import gradio as gr

print(torch.__version__)
print(gr.__version__)
```




