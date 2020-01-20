FROM daangn/faiss:1.5.3

RUN pip install grpcio grpcio-tools

RUN conda install tensorflow
RUN pip install pillow

RUN mkdir -p /app
WORKDIR /app

#ONBUILD COPY requirements.txt /usr/src/app/
#ONBUILD RUN pip install --no-cache-dir -r requirements.txt

RUN pip install gevent

# https://tensorflow.blog/2017/05/12/tf-%EC%84%B1%EB%8A%A5-%ED%8C%81-winograd-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%84%A4%EC%A0%95/
ENV TF_ENABLE_WINOGRAD_NONFUSED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

ENTRYPOINT ["python"]
CMD ["server.py"]

HEALTHCHECK --interval=3s --timeout=2s \
  CMD ls /tmp/status || exit 1

RUN apt-get -qq update && apt-get -qq install wget
RUN mkdir nets && cd nets && \
      wget -q https://github.com/tensorflow/models/raw/master/research/slim/nets/__init__.py && \
      wget -q https://github.com/tensorflow/models/raw/master/research/slim/nets/inception_utils.py && \
      wget -q https://github.com/tensorflow/models/raw/master/research/slim/nets/inception_v4.py

RUN pip install -q scikit-learn
RUN pip install -q scipy
RUN pip install -q boto3
RUN pip install -q click

RUN mv nets /opt/conda/lib/python3.7/site-packages/

# for click library
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY *.py /app/
