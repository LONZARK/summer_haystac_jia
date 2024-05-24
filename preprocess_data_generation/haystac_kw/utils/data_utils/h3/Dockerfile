FROM ubuntu:20.04

# Installing the bare build requirements
RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install cmake make gcc libtool git
# Installing useful tools for development
RUN DEBIAN_FRONTEND=noninteractive apt-get install clang-format cmake-curses-gui lcov doxygen -y

RUN mkdir /home/build
WORKDIR /home/build

RUN git clone https://github.com/uber/h3.git/
WORKDIR h3
RUN git checkout v4.1.0

RUN apt-get install -y cmake
RUN apt-get install ca-certificates gpg wget -y
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' |  tee /etc/apt/sources.list.d/kitware.list >/dev/null
RUN apt-get update
RUN rm /usr/share/keyrings/kitware-archive-keyring.gpg
RUN apt-get install kitware-archive-keyring
RUN apt remove --purge --auto-remove cmake -y
RUN apt-get install cmake -y
RUN apt-get install g++ -y
RUN apt-get install build-essential -y
RUN mkdir build
WORKDIR build
RUN cmake -DCMAKE_BUILD_TYPE=Release ..
RUN make -j 9
RUN make install

