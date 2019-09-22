#! /bin/bash

gcloud ml-engine local train \
    --module-name model.train_task \
    --package-path model \
    --job-dir ${JOB_DIR}  \
    -- \
    --model-version ${VERSION} $1 
