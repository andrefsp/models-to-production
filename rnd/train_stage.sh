#! /bin/bash

MODEL_NAME=$1

COMMIT=$(git log -n1 --pretty="format:%h")

BRANCH_NAME=$(git branch --no-color | grep \* | sed s/\*\ //g)

VERSION=${MODEL_NAME}__${BRANCH_NAME}_${COMMIT}

JOB_DIR=$(realpath ".")/../shared-storage/exports/${MODEL_NAME}/${VERSION}/

mkdir -p ${JOB_DIR}

PYTHONPATH=./:./projects/linear_regression/ JOB_DIR=${JOB_DIR} VERSION=${VERSION}  ./projects/linear_regression/train_local.sh
