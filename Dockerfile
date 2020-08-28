FROM tensorflow/tensorflow:2.2.0-gpu

RUN apt update && apt upgrade -y

RUN apt-get install -y libsndfile1

RUN pip3 install -U soundfile \
                    librosa \
                    python_speech_features \
                    notebook \
                    matplotlib 
                    
RUN pip3 install -U sounddevice
                    
RUN mkdir /root/soundpy/

WORKDIR /root/soundpy/
