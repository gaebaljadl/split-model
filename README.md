# 목표
- pytorch로 resnet18 모델을 여러 파트로 나누고 실행해보기
- 분할했을 때 원본과 동일한 결과를 내는 것 확인하기

## 실행
```
// 이미지 빌드 및 컨테이너 실행 (gpu 사용)
// pytorch 이미지가 거의 7GB라 첫 빌드는 오래걸림
docker build -t splitting .
docker run --rm --gpus all splitting
```

## 참고사항
- GPU 없는 환경에서도 돌아갈 것 같긴 한데 아직 확인은 못해봤음
- 테스트한 환경: Window11 + WSL2 + Docker

## 실행하면 무엇이 나오는가?
1. 나름대로 쪼개본 layer 목록 (입력에 가까운 쪽부터 순서대로 출력)
2. 누군가 제공해준 1000개 샘플 데이터에 대한 정확도 (원본, split 모델 동시에 평가)

## 사용한 테스트 데이터
- ImageNet 데이터가 겁나 커서 순순히 다운받기 너무 싫었음...
- 거의 100GB 된다던데?
- 누군가 1000개 클래스마다 이미지 하나씩 골라놓은 repo가 있길래 그걸 다운받아 썼습니다

## resnet 모델
- resnet18은 너무 작아서 그런지 정확도가 50% 정도밖에 안 나왔음
- resnet50으로 바꾸니 80% 이상!
- [resnet 모델 구조 설명하는 블로그 글](https://jisuhan.tistory.com/71)

## 현재 상황
- ResNet 클래스의 forward() 함수를 참고해서 전체 모델을 21개의 layer로 나눠놓았음.  
pytorch는 forward()에서 정의한대로 계산이 이뤄지므로 이걸 우리가 복붙하듯이 따라하면 됨.
   ```python
   def _forward_impl(self, x: Tensor) -> Tensor:
      # See note [TorchScript super()]
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)

      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.fc(x)

      return x
   ```
- 저장할 때 그냥 list 대신 torch.nn.ModuleList라는 클래스를 사용하라던데, 이유는 잘 모르겠습니다
- split.py에 등장하는 ```sequential_layers = split_resnet(model)``` 이 변수를 사용하시면 됩니다.
- ```sequential_layers```에 들어있는 layer를 하나씩 손수 돌려본 결과 원본 모델과 값이 동일하게 나오는 것을 확인했습니다.
   - assert() 썼는데 끝까지 실행됨
   - 사실 계산 과정이 완전히 동일해서 결과가 달라질 이유가 없기도 함...

## TODO
- 모델 둘로 쪼개고 save & load 테스트해보기