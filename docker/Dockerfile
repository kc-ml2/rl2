FROM nvidia/cuda:10.1-devel-ubuntu18.04

RUN apt-get update -y \
	&& apt-get install -y libopenmpi-dev python3-pip  python3-dev git\
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

COPY . /usr/local/rl2
WORKDIR /usr/local/rl2
RUN pip install -e .
