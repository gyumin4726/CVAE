
# Lightning 설치

[Lightning 홈페이지](https://lightning.ai/docs/pytorch/stable/)를 참고하여 아래와 같이 lightning 패키지를 설치한다.

```sh
python -m pip install lightning
```

## Lightning이 제대로 설치되었는지 확인

python을 구동시킨 후, 다음 코드를 실행해 보도록 하자.
```python
import torch
import lightning as L

print(torch.__version__)
print(L.__version__)
```

버젼 정보가 화면에 출력되면 제대로 설치된 것이다.