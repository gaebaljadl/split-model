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

## 사용한 테스트 데이터
- ImageNet 데이터가 겁나 커서 순순히 다운받기 너무 싫었음...
- 거의 100GB 된다던데?
- 누군가 1000개 클래스마다 이미지 하나씩 골라놓은 repo가 있길래 그걸 다운받아 썼습니다

## 현재 상황
- 아래에 있는 _forward_imple 함수를 그대로 복사한 듯한 코드로 같은 class label이 결과로 나오는걸 확인했음
- layer1, layer2처럼 큰 nn.sequential도 쪼개서 layer 목록에 넣는게 다음 목표

## 지금까지 든 생각
이 모델을 어떻게 나눌지는 resnet.py의 _forward_impl 함수를 보며 고민해봅시다...
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

### resnet 모델
- resnet18은 너무 작아서 그런지 정확도가 50% 정도밖에 안 나왔음
- resnet50으로 바꾸니 80% 이상!
- [resnet 모델 구조 설명하는 블로그 글](https://jisuhan.tistory.com/71)

### 일을 어렵게 만드는 요소
1. forward 함수에서 뭔 짓을 하던 결과만 잘 나오면 pytorch는 좋아함.  
   막 프로그래머한테 그래프를 먼저 만들어달라고 하거나  
   layer list를 제공하라고 강제하지 않음.  
   * model.modules()를 호출하면 뭔가 배열을 주긴 하는데 이대로는 사용 불가.  
     일단 돌려보니 결과가 이상했고, 함수 설명 보니까 내부적으로  
     여러 번 사용한 layer는 한 번만 나온다고 하니 좋은 결과를 기대하기 힘들다.
2. 그렇다보니 오히려 우리 입장에선 약간 곤란해짐.  
   안에서 어떻게 돌아가는지 forward 함수를 뜯어보기 전까진 아무것도 알 수가 없음.  
   모델을 분할하고 싶으면 모델의 구조와 연산 순서를 모두 파악해야 함.

### 앞으로 알아봐야 할 정보
1. modules()말고 제대로된 layer 배치?를 얻어낼 방법이 혹시나 존재하는지?
   - 예를 들어, 우리가 수동으로 nn.sequential을 풀어서 layer 배열을 만든다던가?
   - 보니까 residual connection이 포함된 bottleneck이라 부르는 블록이 있는데,  
   이걸 분할 가능한 최소 단위로 삼아서 layer list를 만들 수 있을 것 같음
2. pytorch tensor를 네트워크 상에서 어떻게 전달할 것인지? (인코딩 디코딩 요런거)
3. layer 목록 가져와서 한 단계씩 넣어보는걸 했는데 뭔가 메모리 누수가 일어나는 것 같음.   
어떻게 확인할진 모르겠음.
