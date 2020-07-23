FROM ubuntu:18.04

RUN apt update && apt upgrade -y

RUN apt install -y python3-minimal \ 
                python3-pip 
                
RUN mkdir /root/pysoundtool/

WORKDIR /root/pysoundtool/

COPY ./requirements.txt . 

RUN pip3 install -r requirements.txt

RUN pip3 install notebook

#CMD /bin/bash 
CMD jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root 



