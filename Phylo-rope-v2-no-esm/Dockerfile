# Using an official Python runtime with CUDA support as a parent image (https://hub.docker.com/r/nvidia/cuda/)
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel as base

ENV HOST docker
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y git

# Disable pip cache: https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
ENV PIP_NO_CACHE_DIR=1
#install from requirements.txt
COPY requirements.txt /home/requirements.txt
#apt get git
RUN pip install --no-cache-dir -r /home/requirements.txt

WORKDIR /home