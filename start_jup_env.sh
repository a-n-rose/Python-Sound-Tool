docker run -it --rm \
            --gpus all \
            --privileged=true \
            -v "$PWD":"/root/soundpy/" \
            -p 8888:8888 aju
            #-v "/audiodir/data":"/root/soundpy/data" \
