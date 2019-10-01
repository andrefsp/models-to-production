#! /bin/bash

docker rmi ${MODEL_NAME}

model_export_path=$(realpath --relative-to=. ${EXPORT_PATH})

docker build --no-cache -t ${MODEL_NAME} --build-arg MODEL_NAME=${MODEL_NAME} --build-arg EXPORT_PATH=${model_export_path}  . 


docker run --rm -it -p 9000:9000 ${MODEL_NAME}
