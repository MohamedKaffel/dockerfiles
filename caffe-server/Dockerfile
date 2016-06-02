FROM kaixhin/caffe
MAINTAINER Haoming Wang <haoming.exe@gmail.com>

WORKDIR /root/caffe
RUN sudo apt-get install wget
RUN rm -rf ~/.matplotlib/fontList.cache
RUN ./scripts/download_model_binary.py models/bvlc_reference_caffenet
RUN ./data/ilsvrc12/get_ilsvrc_aux.sh
RUN pip install -r examples/web_demo/requirements.txt
EXPOSE 5000
CMD python examples/web_demo/app.py
