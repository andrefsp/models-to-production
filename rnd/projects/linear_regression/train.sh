#! /bin/bash

gcloud ml-engine jobs submit training ${VERSION} \
        --module-name model.train_task \
        --package-path model \
        --config config.yaml \
        --job-dir ${JOB_DIR} \
        --stream-logs \
        -- \
        --model-version ${VERSION} 
