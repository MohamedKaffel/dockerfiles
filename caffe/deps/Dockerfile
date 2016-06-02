# Start with Ubuntu base image
FROM ubuntu:14.04
MAINTAINER Haoming Wang <haoming.exe@gmail.com>

# Install git, bc and dependencies
RUN apt-get update && apt-get install -y \
  git \
  bc \
  cmake \
  libatlas-base-dev \
  libatlas-dev \
  libboost-all-dev \
  libopencv-dev \
  libprotobuf-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  protobuf-compiler \
  libhdf5-dev \
  libleveldb-dev \
  liblmdb-dev \
  libsnappy-dev \
  python-dev \
  python-pip \
  python-numpy \
  wget \
  gfortran > /dev/null

# Clone Caffe repo and move into it
RUN cd /root && git clone https://github.com/BVLC/caffe.git && cd caffe && \
# Install Anaconda2
 wget --quiet https://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh && \
    /bin/bash Anaconda2-4.0.0-Linux-x86_64.sh -b -p /opt/anaconda2 && \
    rm Anaconda2-4.0.0-Linux-x86_64.sh 
ENV PATH /opt/anaconda2/bin:$PATH
# Install python dependencies
RUN cat python/requirements.txt | xargs -n1 pip install
