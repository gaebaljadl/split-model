from flask import Flask, request, jsonify, Response
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
import numpy as np
from splitter import *

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
# numpy array 형태로 오니까 to_tensor만 적용해주면 됨.
is_request_from_client = True

# 이미지 전처리를 위한 변환 정의
to_tensor = transforms.ToTensor()
preprocess_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    to_tensor,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 입력 이미지를 전처리하고 모델에 전달하여 예측을 생성하는 함수
def predict_image(input) -> torch.Tensor:
    with torch.no_grad():
        output = split_model(input)
    return output


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


@app.route('/predict', methods=['POST'])
def predict():
    # 입력 이미지 받기.
    # 클라이언트가 보낸 요청인 경우는 진짜 이미지 파일이고,
    # 일부분만 담당하는 파트에서 보내온 '나머지 연산' 요청인 경우는 numpy array임.
    file = request.files['file']

    if is_request_from_client:
        # 변환 과정: 바이트 배열 -> 이미지 파일 -> 전처리 -> torch.Tensor
        image = Image.open(io.BytesIO(file.read()))
        input = preprocess_image(image).unsqueeze(0).to(device)
        print('request from client')
    else:
        # 변환 과정: 바이트 배열 -> numpy array -> shape 조정 -> torch.Tensor
        shape = request.json['shape']
        print('request from previous split model with shape: {}'.format(shape))
        # shape = [1, 512, 28, 28]
        numpy_array = np.frombuffer(file.read(), dtype=float)
        np.reshape(numpy_array, shape)
        input = to_tensor(numpy_array).to(device)

    # 예측 생성
    output = predict_image(input)

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
        # 변환 과정: gpu tensor -> cpu tensor -> numpy array -> byte array
        output_shape = output.shape
        output_bytes = output.to('cpu').numpy().tobytes()
        byte_stream = io.BytesIO(output_bytes)

        # 다음 파트를 담당하는 pod에 나머지 연산 요청
        print('forwarding request to {} with intermediate output shape: {}...'.format(next_model_addr, output_shape))
        res = requests.post('http://{}/predict'.format(next_model_addr), json={'shape': output_shape}, files={'file': byte_stream})

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
