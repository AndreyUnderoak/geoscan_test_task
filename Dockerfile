FROM tensorflow/tensorflow

SHELL ["/bin/bash", "-ci"]


RUN apt update && \
    # Install build tools, build dependencies and python
    apt install -y python3-opencv

RUN pip install opencv-python
RUN pip install matplotlib
	
WORKDIR /task_workspace