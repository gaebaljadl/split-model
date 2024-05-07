from flask import Flask, request, jsonify, Response
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
import numpy as np
import json
from splitter import *

# Overview:
# app.py는 원본 모델의 전체 또는 일부 레이어에 해당하는 연산을 처리하는 stateful 서버임.
#
# 서버는 두 개의 엔드포인트로 state 변경 및 인퍼런스 요청을 처리함
# 1. /configure - GET 요청으로 처리할 레이어 범위 및 다음 서버를 지정
# 2. /predict - POST로 수신한 이미지 파일 또는 텐서를 디코딩하고 자신이 담당하는 레이어 범위의 연산을 수행
#
# 모델의 분할은 다음과 같이 이뤄짐
# 1. 원본 모델을 쪼갤 수 있는 만큼 쪼개서 레이어 리스트로 변환 (module_list 변수)
# 2. 처리할 레이어 범위를 갖고 torch.nn.Sequential으로 partial model을 생성 (split_model 변수)
# 3. /configure 요청이 들어올 때마다 module_list를 사용해 새로운 모델을 만들고 split_model에 저장
#
# 예를 들어, 전체 모델을 L1, L2, L3로 나누었다면 데이터의 흐름은 다음과 같음
# 유저 -> L1 -> L2 -> L3 (여기서 인퍼런스 완료, response 전송 시작) -> L2 -> L1 -> 유저
app = Flask(__name__)

# GPU 사용 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print('The server is using', device, 'to perform inference')

# ResNet 불러오기
original_model = load_pretrained_resnet()
module_list = convert_to_module_list(original_model)
split_model = torch.nn.Sequential(*module_list).to(device)
split_model.eval()

# 다음 레이어가 남아있는 경우 요청을 보낼 주소
next_model_addr = 'None'

# 첫 레이어를 담당하는 경우 요청을 유저가 보냄.
# 이미지 파일이 들어오니까 전처리(preprocess_image) 필요.
# False인 경우는 바로 전 파트의 output tensor가
# numpy array 형태로 오니까 torch.from_numpy() 적용해주면 됨.
is_request_from_client = True

# 이미지 전처리를 위한 변환 정의
preprocess_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 입력 이미지를 전처리하고 모델에 전달하여 예측을 생성하는 함수
def predict_image(input) -> torch.Tensor:
    with torch.no_grad():
        output = split_model(input)
    return output


# 서버에서 담당할 레이어의 범위와 다음 서버의 주소를 설정하는 엔드포인트.
# 쿼리에 세 가지 값을 넣어줘야 한다
# 1. start - 레이어 범위의 시작 인덱스
# 2. end - 레이어 범위의 마지막 인덱스 + 1 ([start, end)라는 half open interval 생각하면 됨)
# 3. nextaddr - 다음 서버의 주소. 이 서버가 모델의 마지막 레이어를 포함하는 경우 주소 대신 None을 사용.
@app.route('/configure', methods=['GET'])
def configure():
    layer_start = request.args.get('start', type = int)
    layer_end = request.args.get('end', type = int)

    global is_request_from_client
    is_request_from_client = layer_start == 0

    global next_model_addr
    next_model_addr = request.args.get('nextaddr', type=str)

    global split_model
    split_model = torch.nn.Sequential(*module_list[layer_start:layer_end]).to(device)
    split_model.eval()
    return 'layer_start: {}, layer_end: {}, next_model_addr: {}'.format(layer_start, layer_end, next_model_addr)


# 인퍼런스 요청을 처리하는 엔드포인트.
# 입력과 출력에서 각각 두 가지 경우가 존재한다
# 1. 입력
# case 1) 유저가 보낸 이미지를 받은 경우 이미지 전처리를 수행
# case 2) 이전 파트를 담당하는 서버가 보낸 텐서를 받은 경우 전처리 없이 바로 torch.Tensor로 복원
# 2. 출력
# case 1) 전체 모델의 연산이 끝난 경우 확률이 가장 높은 클래스의 인덱스를 json 형태로 반환
# case 2) 모델의 일부분만 계산된 경우 다음 서버로 자신의 연산 결과 텐서를 전송하고, 받은 결과물을 요청한 곳에 forward
@app.route('/predict', methods=['POST'])
def predict():
    if is_request_from_client:
        # 변환 과정: 바이트 배열 -> 이미지 파일 -> 전처리 -> torch.Tensor
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        input = preprocess_image(image).unsqueeze(0).to(device)
    else:
        # 변환 과정: 바이트 배열 -> numpy array -> shape 조정 -> torch.Tensor
        print('start reading output tensor from previous pod...', flush=True)
        shape_json_bytes = request.files['shape_json']
        shape = json.loads(shape_json_bytes.read())['shape']
        print('request from previous split model with shape: {}'.format(shape), flush=True)

        output_bytes = request.files['output_bytes']
        numpy_array = np.frombuffer(output_bytes.read(), dtype=np.float32)
        numpy_array = np.reshape(numpy_array, shape)
        input = torch.from_numpy(numpy_array).to(device)

    # 예측 생성 (GPU 연산 가능성도 있으니 cpu 메모리로 이동)
    output = predict_image(input).to('cpu')

    # Case 1) 이번 파트가 최종 출력을 계산했음
    if next_model_addr == 'None':
        # 결과를 리스트로 변환
        output_list = output.tolist()

        # 가장 큰 값의 인덱스 찾기
        max_index = output_list[0].index(max(output_list[0]))

        # 결과 반환
        return jsonify({'predicted_class': max_index}), 200
    # Case 2) 이번 파트는 일부분만 계산했고, 나머지 연산은 다음 파트로 넘어가야 함
    else:
        # torch.Tensor를 바이트로 변환하기.
        # 차원 구분이 없는 바이트 배열로 보내다보니
        # 받는 쪽에서 복원할 때 shape 정보 필요함!
        # 변환 과정: tensor -> numpy array -> byte array
        # 참고: numpy()로 변환하면 타입이 np.float32임
        #       np.frombuffer()는 디폴트 타입이 np.float64라서 반드시 타입을 명시해야 함!!!
        output_shape = list(output.shape)
        output_bytes = output.numpy().tobytes()

        # 다음 파트를 담당하는 pod에 나머지 연산 요청.
        # multipart post 보내기: https://stackoverflow.com/questions/35939761/how-to-send-json-as-part-of-multipart-post-request#comment59538358_35939761
        shape_json = json.dumps({'shape': output_shape})
        files = {
            'shape_json': ('shape_json', shape_json, 'application/json'),
            'output_bytes': ('output_bytes', output_bytes, 'application/octet-stream')
        }
        print('forwarding request to {} with intermediate output shape: {}, type: {}...'.format(next_model_addr, output_shape, output.type()), flush=True)
        res = requests.post('http://{}/predict'.format(next_model_addr), files=files)

        # response forwarding: https://stackoverflow.com/questions/6656363/proxying-to-another-web-service-with-flask
        #region exlcude some keys in :res response
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']  #NOTE we here exclude all "hop-by-hop headers" defined by RFC 2616 section 13.5.1 ref. https://www.rfc-editor.org/rfc/rfc2616#section-13.5.1
        headers          = [
            (k,v) for k,v in res.raw.headers.items()
            if k.lower() not in excluded_headers
        ]
        #endregion exlcude some keys in :res response

        response = Response(res.content, res.status_code, headers)
        return response, 200
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
