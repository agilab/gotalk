FROM ubuntu:16.04

RUN mkdir /gotalk

ADD libtensorflow.so /usr/lib/libtensorflow.so
ADD frozen_model.pb /gotalk/frozen_model.pb
ADD vocabulary.txt /gotalk/vocabulary.txt
ADD server /gotalk/server

EXPOSE 80

CMD /gotalk/server --vocab /gotalk/vocabulary.txt --model /gotalk/frozen_model.pb --port 80
