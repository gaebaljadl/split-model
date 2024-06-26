# Deploying Splitted Deep Learning Model to Kubernetes Cluster

## 목표

### 완료된 목표

- pytorch로 resnet50 모델을 여러 파트로 나누고 실행해보기
- 분할했을 때 원본과 동일한 결과를 내는 것 확인하기
- 웹서버로 감싸 레이어 범위 및 다음 파트 담당하는 서버 ip주소 지정하기
- 여러 서버로 분할된 환경에서 inference 잘 진행되는지 확인하기
- 쿠버네티스 서비스의 cluster IP로 통신하기
- ConfigMap으로 /configure 엔드포인트 대체하기

### TODO

- 로드밸런서 만들기

## 실행

### 로컬

모델 서버를 두 개의 도커 컨테이너로 실행

```sh
# 이미지 빌드 및 컨테이너 실행 (gpu 사용, 포트는 어디로 매핑될지 모름...)
# pytorch 이미지가 거의 7GB라 첫 빌드는 오래걸림
docker build -t splitting .
docker run -d -p 33333:8080 --rm --gpus all splitting
docker run -d -p 44444:8080 --rm --gpus all splitting
```

테스트

```sh
# 모델의 레이어 범위 및 다음 서버 주소 지정.
curl -G -d "start=0" -d "end=10" -d "nextaddr=host.docker.internal:44444" http://localhost:33333/configure

# 마지막 파트에는 "nextaddr=None" 넘겨주기.
curl -G -d "start=10" -d "end=21" -d "nextaddr=None" http://localhost:44444/configure

# Model inference ({"predicted_class":436}이라고 나오면 성공)
curl -X POST -F "file=@./input/20231020_01110305000006_L00.jpg" http://localhost:33333/predict
```

### 쿠버네티스 환경

1. 도커 빌드 및 이미지 푸시 (현재 docker hub에 업로드된 상태라 스킵하면 됨)

```sh
docker build -t ownfos/splitting . # <docker-hub-id>/<image>
docker push ownfos/splitting
```

2. 쿠버네티스 클러스터에 Deployment 배포 및 NodePort Service 생성

```sh
kubectl apply -f deploy/splitting-configmap.yaml
kubectl apply -f deploy/splitting-deployment.yaml
kubectl apply -f deploy/splitting-service.yaml
```

3. NodePort Service의 클러스터 IP 확인
```sh
kubectl get services
```
![클러스터 IP 확인](screenshots/getclusterip.png)

4. 테스트
   1. ConfigMap을 사용하지 않은 경우 수동으로 /configure 엔드포인트에 설정값 전달  
      ```sh
      # 용어 정리:
      # splitting1에 접근하는 주소 addr1 = splitting1-service의 클러스터 IP
      # splitting2에 접근하는 주소 addr2 = splitting2-service의 클러스터 IP
      # * 예시로 주어진 service들은 기본 포트인 80을 사용하기 때문에 포트 번호는 생략 가능

      # 모델의 레이어 범위 및 다음 서버 주소 지정.
      curl -G -d "start=0" -d "end=10" -d "nextaddr=addr2" http://addr1/configure

      # 마지막 파트에는 "nextaddr=None" 넘겨주기.
      curl -G -d "start=10" -d "end=21" -d "nextaddr=None" http://addr2/configure
      ```

   2. 인퍼런스 요청
      ```sh
      # Model inference ({"predicted_class":436}이라고 나오면 성공)
      curl -X POST -F "file=@./input/20231020_01110305000006_L00.jpg" http://addr1/predict
      ```

<!-- TODO -->

#### [참고] 서버는 두 가지 방법으로 설정 가능
1. 서버를 실행한 뒤 /configure 엔드포인트 사용
   - 로컬 환경이나 도커처럼 IP와 포트를 확인할 수 있는 경우에 가능
   
2. 환경변수 LAYER_START, LAYER_END, NEXTADDR 설정하고 서버 실행
   - k8s 클러스터처럼 특정 포트에 접근하기가 어려운 경우 사용
   - ConfigMap을 사용하면 pod 생성 시 환경변수를 주입할 수 있음

## test.py 동작 과정

1. 이미지를 빌드할 때 splitter.py를 실행해서 pretrained resnet 모델을 다운받고  
이를 둘로 쪼갠 head와 tail을 각각 head.pth, tail.pth 파일에 저장함.
2. 컨테이너를 실행하면 원본 모델, head, 그리고 tail을 불러옴.
3. head와 tail의 레이어 정보를 출력함.
4. 누군가 제공해준 1000개 샘플 데이터를 넣어보고 정확도를 출력함.
   - 원본과 split model이 같은 결과를 내는지 assert()로 체크하기 때문에  
   실행이 정상적으로 끝났다면 정확도 문제 x

### 주의사항

- 메모리 요구량이 상당해서 실행하다가 메모리 부족으로 실패할 수 있음

## Wrapper 웹서버 동작 과정

1. 서버를 실행한 뒤 /configure 엔드포인트에 start, end, nextaddr 세 개의 파라미터로 모델 설정
2. 요청이 들어오면 files에 담긴 바이너리 데이터를 이미지로 변환
3. 자신이 담당하는 파트를 계산한 뒤, 출력 텐서를 바이너리 형태로 다음 파트를 담당하는 서버에게 전송
   - GPU 텐서 -> CPU 텐서 -> numpy array -> 바이너리
4. 다음 파트를 담당하는 서버는 이를 다시 텐서로 복원
5. 최종적으로 모든 연산을 마치면 prediction class index를 json 형태로 반환

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
- resnet50으로 바꾸니 90% 이상!
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

## torch.Tensor 전송하기 (인코딩, 디코딩)

- [참고자료](https://stackoverflow.com/questions/70174676/how-to-send-an-numpy-array-or-a-pytorch-tensor-through-http-post-request-using-r)
