export PATH=${PATH}:${GOPATH}/bin

mkdir -p vendor
PROTOC_OPTS='-I tensorflow -I serving --go_out=plugins=grpc:vendor'

eval "protoc $PROTOC_OPTS serving/tensorflow_serving/apis/*.proto"
eval "protoc $PROTOC_OPTS serving/tensorflow_serving/config/*.proto"
eval "protoc $PROTOC_OPTS serving/tensorflow_serving/util/*.proto"
eval "protoc $PROTOC_OPTS serving/tensorflow_serving/sources/storage_path/*.proto"
eval "protoc $PROTOC_OPTS tensorflow/tensorflow/core/framework/*.proto"
eval "protoc $PROTOC_OPTS tensorflow/tensorflow/core/example/*.proto"
eval "protoc $PROTOC_OPTS tensorflow/tensorflow/core/lib/core/*.proto"
eval "protoc $PROTOC_OPTS tensorflow/tensorflow/core/protobuf/{saver,meta_graph}.proto"
