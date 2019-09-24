#! /bin/bash

PWD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

arch=`uname | tr 'A-Z' 'a-z'`

args=$@

if [ -z "${args-unset}" ]; then
    args="./... -tags=integration"
fi

export C_INCLUDE_PATH=${C_INCLUDE_PATH}:${PWD_DIR}/libs/ldpath/${arch}/include/
export LIBRARY_PATH=$LIBRARY_PATH:${PWD_DIR}/libs/ldpath/${arch}/lib/

if [ "${arch}" == "darwin" ]; then
	export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${PWD_DIR}/libs/ldpath/${arch}/lib/
elif [ "${arch}" == "linux" ]; then
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD_DIR}/libs/ldpath/${arch}/lib/
else
	echo "::::::::: Architecture not supported ::::::::::"
	exit 1
fi

go build . && ./demo
