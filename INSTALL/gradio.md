# Gradio 설치

[Gradio 홈페이지](https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment)를 참고하여 아래와 같이 gradio 패키지를 설치한다.
```sh
pip install gradio
```

## Gradio가 제대로 설치되었는지 확인

python을 구동시킨 후, 다음 코드를 실행해 보자. 
```python
import torch
import gradio as gr

print(torch.__version__)
print(gr.__version__)
```

에러 없이 버젼 정보가 출력되면 제대로 설치된 것이다.
