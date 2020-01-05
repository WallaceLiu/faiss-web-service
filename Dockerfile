ARG IMAGE
FROM ${IMAGE}

COPY requirements.txt /Users/liuning11/project/faiss-web-service/requirements.txt
RUN conda install -y -c conda-forge --file /Users/liuning11/project/faiss-web-service/requirements.txt

COPY bin /Users/liuning11/project/faiss-web-service/bin
COPY src /Users/liuning11/project/faiss-web-service/src
COPY resources /Users/liuning11/project/faiss-web-service/resources

WORKDIR /Users/liuning11/project/faiss-web-service

ENTRYPOINT ["/Users/liuning11/project/faiss-web-service/bin/faiss_web_service.sh"]
