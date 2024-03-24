# 목표
- pytorch로 resnet50 모델을 여러 파트로 나누고 실행해보기
- 분할했을 때 원본과 동일한 결과를 내는 것 확인하기

## 실행
```
// 이미지 빌드 및 컨테이너 실행 (gpu 사용)
// pytorch 이미지가 거의 7GB라 첫 빌드는 오래걸림
docker build -t splitting .
docker run --rm --gpus all splitting

// vscode 등으로 백그라운드 컨테이너에 접근하고 싶은 경우
docker run -dt --rm --gpus all splitting bash
```

## 참고사항
- GPU 없는 환경에서도 돌아갈 것 같긴 한데 아직 확인은 못해봤음
- 테스트한 환경: Window11 + WSL2 + Docker

## 동작 과정
1. 이미지를 빌드할 때 splitter.py를 실행해서 pretrained resnet 모델을 다운받고  
이를 둘로 쪼갠 head와 tail을 각각 head.pth, tail.pth 파일에 저장함.
2. 컨테이너를 실행하면 원본 모델, head, 그리고 tail을 불러옴.
3. head와 tail의 레이어 정보를 출력함.
4. 누군가 제공해준 1000개 샘플 데이터를 넣어보고 정확도를 출력함.
   - 원본과 split model이 같은 결과를 내는지 assert()로 체크하기 때문에  
   실행이 정상적으로 끝났다면 정확도 문제 x

## 사용한 테스트 데이터
- ImageNet 데이터가 겁나 커서 순순히 다운받기 너무 싫었음...
- 거의 100GB 된다던데?
- 누군가 1000개 클래스마다 이미지 하나씩 골라놓은 repo가 있길래 그걸 다운받아 썼습니다
   - ```TestDataset```의 생성자 참고
- 테스트 데이터에 라벨은 포함되지 않아서 이미지 이름으로 클래스 인덱스를 알아내야 했는데,  
다행히 클래스 이름이랑 클래스 인덱스를 dict 형식으로 제공하는 repo가 또 있어서 그걸 활용함.
   - ```load_class_index_to_name_dict()``` 참고

## resnet 모델
- resnet18은 너무 작아서 그런지 정확도가 50% 정도밖에 안 나왔음
- resnet50으로 바꾸니 80% 이상!
- [resnet 모델 구조 설명하는 블로그 글](https://jisuhan.tistory.com/71)

## 그래서 모델은 어떻게 나눈건데?
- pytorch는 모델의 forward() 함수를 그대로 실행하니까, 우리가 이걸 복붙하듯이 따라하면 됨.
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
- 위 코드에서 layer1 ~ layer4는 nn.Sequential이어서 더 세부적으로 쪼갤 수 있음.  
list 순회하듯이 iterate => 배열에 추가.
   ```python
   def convert_to_module_list(resnet_model: ResNet) -> list[torch.nn.Module]:
      """
      Extract modules (i.e. layers) from the ResNet model
      and put them into a list
      """
      layers = []

      layers.append(resnet_model.conv1)
      layers.append(resnet_model.bn1)
      layers.append(resnet_model.relu)
      layers.append(resnet_model.maxpool)

      # These layers are nn.Sequential, so we should iterate over them.
      for layer in resnet_model.layer1:
         layers.append(layer)
      for layer in resnet_model.layer2:
         layers.append(layer)
      for layer in resnet_model.layer3:
         layers.append(layer)
      for layer in resnet_model.layer4:
         layers.append(layer)
      
      # Output layer utilizes torch.flatten(),
      # so it was inevitable to encapsulate
      # the last layers into a custom module.
      layers.append(ResNetOutputLayer(resnet_model.avgpool, resnet_model.fc))

      return layers
   ```