[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

dockerfiles
===========

Compilation of Dockerfiles with automated builds enabled on the [Docker Hub](https://hub.docker.com/u/flyingmouse/). **Not suitable for production environments.** These images are under continuous development, so breaking changes may be introduced.

Nearly all images are based on Ubuntu Core 14.04 LTS, built with minimising size/layers and [best practices](https://docs.docker.com/articles/dockerfile_best-practices/) in mind.

Up-to-date builds
-----------------

Some builds based on certain software have builds that are triggered on schedule via a cron script to stay up to date on a weekly basis. These are:

- [Caffe](https://github.com/BVLC/caffe)

Daemonising containers
----------------------

Most containers run as a foreground process. To daemonise (in Docker terminology, detach) such a container it is possible to use:

`docker run -d <image> sh -c "while true; do :; done"`

It is now possible to access the daemonised container, for example using bash:

`docker exec -it <id> bash`

Caffe
----

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

This Docker image Caffe is compiled with OpenCV3 and Anaconda support

Webdemo
----

The demo server requires Python with some dependencies. To make sure you have the dependencies, please run

`pip install -r examples/web_demo/requirements.txt`

and also make sure that you’ve compiled the Python Caffe interface and that it is on your PYTHONPATH

Use Caffe Web Demo Docker:

	docker pull flyingmouse/caffe-ocr
	docker run -p 80:5000 -d flyingmouse/caffe-ocr

Automated Builds
----------------

[Automated Builds](https://docs.docker.com/docker-hub/builds/) on the Docker Hub have several advantages, including reproducibility and security. However the build cluster has the following limits for Automated Builds:

- 2 hours
- 1 CPU
- 2 GB RAM
- 512 MB swap
- 30 GB disk space

The main tip for keeping within the CPU and memory limits is to reduce parallelism/forking processes. Due to their logging system, redirecting stdout/stderr to /dev/null can potentially save a reasonable amount of memory.

