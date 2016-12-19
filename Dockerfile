FROM ubuntu:14.04

RUN apt-get -y update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:george-edison55/cmake-3.x
RUN apt-get -y update 
RUN apt-get install -y python3 python3-dev python3-pip build-essential \
	zlib1g-dev libsdl2-dev libjpeg-dev nasm tar libbz2-dev libgtk2.0-dev \
	cmake git libfluidsynth-dev libgme-dev libopenal-dev timidity \
	libwildmidi-dev libboost-all-dev wget zip
RUN pip3 install numpy
RUN pip3 install git+https://github.com/Marqt/ViZDoom
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow

CMD cd /src && python3 network.py
