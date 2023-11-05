FROM ubuntu

RUN apt-get update &&\
    apt-get install -y git wget tar cmake g++ apt-utils libblas-dev libopenblas-dev \
    libopenblas-base libopenblas0 liblapack-dev liblapack3 unzip

ARG BASE_DIRECTORY=/usr/local/stitching

WORKDIR ${BASE_DIRECTORY}

# Compile and install levmar
RUN wget https://users.ics.forth.gr/~lourakis/levmar/levmar-2.6.tgz &&\
    tar -xf levmar-2.6.tgz &&\
    mkdir -p ${BASE_DIRECTORY}/levmar-2.6/build 

WORKDIR ${BASE_DIRECTORY}/levmar-2.6/build

RUN cmake ../ -DCMAKE_CXX_STANDARD_LIBRARIES="-lm" -D BUILD_DEMO=FALSE &&\
    make

RUN echo $pwd && echo `ls`
RUN cp liblevmar.a /usr/local/lib/ &&\
    mkdir -p /usr/local/lnclude/levmar &&\
    cp ../levmar.h /usr/local/lnclude/levmar/

WORKDIR ${BASE_DIRECTORY}

# Compile and install opencv
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip &&\
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip &&\
    unzip opencv.zip &&\
    unzip opencv_contrib.zip &&\
    mkdir -p build_opencv

WORKDIR ${BASE_DIRECTORY}/build_opencv

RUN cmake -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x &&\
    cmake --build . --config Release -- -j16 &&\
    make install

# Install stitching code
WORKDIR ${BASE_DIRECTORY}
RUN git clone https://github.com/nbubis/AutoStitching
RUN mkdir -p ${BASE_DIRECTORY}/AutoStitching/build
WORKDIR ${BASE_DIRECTORY}/AutoStitching/build
RUN cmake ../ && make
