FROM flyingmouse/caffe
MAINTAINER Haoming Wang <haoming.exe@gmail.com>

WORKDIR /root/caffe
RUN rm -rf ~/.cache/matplotlib
RUN ./data/ilsvrc12/get_ilsvrc_aux.sh
ADD app.py /root/caffe/examples/web_demo/app.py
ADD index.html /root/caffe/examples/web_demo/templates/index.html
RUN mkdir -p /root/ocr/googlenet
ADD googlenet/train_val.caffemodel /root/ocr/googlenet
ADD googlenet/deploy.prototxt /root/ocr/googlenet
ADD googlenet/synset_words.txt /root/ocr/googlenet
RUN pip install -U pip && \
  pip install -r examples/web_demo/requirements.txt && \
  pip install protobuf
EXPOSE 5000
CMD python examples/web_demo/app.py
