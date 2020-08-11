FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN apt update && apt upgrade -y

RUN apt-get install -y libsndfile1

RUN pip3 install -U soundfile \
                    librosa \
                    python_speech_features \
                    notebook \
                    matplotlib
                    
RUN mkdir /root/soundpy/

WORKDIR /root/soundpy/
