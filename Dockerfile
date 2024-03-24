# vanilla pytorch 컨테이너 실행 명령: docker run -dt --rm --gpus all pytorch/pytorch

FROM pytorch/pytorch

RUN apt-get update && apt-get install -y git

RUN mkdir /test; cd /test;\
    # 클래스마다 이미지 한 장씩 선정한 repo
    git clone https://github.com/EliSchwartz/imagenet-sample-images.git;\
    # 클래스 인덱스를 이름으로 매핑하는 txt파일
    git clone https://gist.github.com/942d3a0ac09ec9e5eb3a.git

COPY ./splitter.py /test/splitter.py
COPY ./sample_data.py /test/sample_data.py

# 모델 로딩하는 코드만 먼저 돌려서 이미지 빌드할 때 pretrained weight 다운받아놓기
RUN cd /test; python splitter.py

COPY ./test.py /test/test.py

WORKDIR /test

CMD ["python", "test.py"]