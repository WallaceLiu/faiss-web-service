# faiss-server
faiss similarity search server

## Requirements
* Docker
* Python

### Settings
```
$ cp config.sample.sh config.sh
$ vi config.sh
```

### Commands
Docker 서버 실행
```
$ ./run [faiss-server name]
```

Docker 서버 중지
```
$ ./stop [faiss-server name]
```

Docker 서버 재시작
```
$ ./restart [faiss-server name]
```

클라이언트 요청 테스트
```
$ ./client [faiss-server name]
```

Train and save index file
```
docker run --rm -it daangn/faiss-server train.py [options]
```
