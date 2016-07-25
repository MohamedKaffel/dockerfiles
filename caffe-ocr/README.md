[![Docker Pulls](https://img.shields.io/docker/pulls/flyingmouse/caffe-ocr.svg)](https://hub.docker.com/r/flyingmouse/caffe-ocr/)
[![Docker Stars](https://img.shields.io/docker/stars/flyingmouse/caffe-ocr.svg)](https://hub.docker.com/r/flyingmouse/caffe-ocr/)

Caffe Web Demo
=====
Ubuntu Core 14.04 + [Caffe](http://caffe.berkeleyvision.org/). Includes Python interface.

Usage
-----
Use Caffe Web Demo Docker:
	docker build -t webdemo:cpu .
	docker run -p 80:5000 -d <image>

For more information on Caffe Web Demo on Docker, see the [repo readme](https://github.com/flyingmouse/dockerfiles#Webdemo).