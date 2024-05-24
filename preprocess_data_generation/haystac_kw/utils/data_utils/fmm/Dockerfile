FROM ubuntu:focal

RUN  apt-get update && apt-get install -y software-properties-common
RUN  add-apt-repository ppa:ubuntugis/ppa -y
RUN  apt-get -q update -y
RUN  apt-get install libboost-dev libboost-serialization-dev gdal-bin libgdal-dev make cmake libbz2-dev libexpat1-dev swig python-dev git -y
RUN apt-get install build-essential -y
RUN git clone https://github.com/cyang-kth/fmm.git
WORKDIR ./fmm

RUN mkdir build
WORKDIR ./build
RUN cmake ..
RUN make -j4
RUN make install
RUN mkdir /home/workspace
COPY fmm.xml /home/fmm.xml
COPY ubodt.xml /home/ubodt.xml