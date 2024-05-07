FROM pytorch/pytorch

RUN apt-get update && apt-get install -y git iputils-ping net-tools

# pytorch 이미지에 Flask는 없으니 설치
RUN pip install Flask

RUN mkdir /test; cd /test;\
    # 클래스마다 이미지 한 장씩 선정한 repo
    git clone https://github.com/EliSchwartz/imagenet-sample-images.git;\
    # 클래스 인덱스를 이름으로 매핑하는 txt파일
    git clone https://gist.github.com/942d3a0ac09ec9e5eb3a.git

COPY ./splitter.py /test/splitter.py
COPY ./sample_data.py /test/sample_data.py

# 모델 로딩하는 코드만 먼저 돌려서 이미지 빌드할 때 pretrained weight 다운받아놓기
# test.py에서 사용하는 head.pth와 tail.pth도 여기서 생성된다
RUN cd /test; python splitter.py

# 여기는 수정이 자주 일어나니 나중에 복사
COPY ./test.py /test/test.py
COPY ./app.py /test/app.py

WORKDIR /test

# app.py에서 8080포트로 웹서버 실행할 예정
EXPOSE 8080

CMD ["python", "app.py"]