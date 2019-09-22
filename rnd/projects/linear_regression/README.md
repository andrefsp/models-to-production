## Linear model


# Train model locally

To train the linear model using gcloud

    $ gcloud ml-engine local train --module-name linear.train_task --package-path linear  -- --train-file linear/data/linear.train.csv --export-path ./linear/serve/

It will save the model on a `./linear/serve/` folder.

or using gcloud storage

    $ gcloud ml-engine local train --module-name linear.train_task --package-path linear  -- --train-file gs://ai-learning/models/linear/data/linear.train.csv --export-path gs://ai-learning/models/linear/serve/
   
 It will save the model on a `gs://ai-learning/models/linear/serve/` folder.


# Train on the cloud.

    $ gcloud ml-engine jobs submit training linearV0_0 --module-name linear.train_task --package-path linear --job-dir gs://ai-learning/models/linear/v0/job/  -- --train-file gs://ai-learning/models/linear/data/linear.train.csv  --export-path gs://ai-learning/models/linear/v0/export/


# Making predictions

You can than run predictions using `saved_model_cli`

    $ saved_model_cli run --dir ./linear/serve/ --tag_set serve --signature_def serving_default --input_exprs='X=np.float32(4)'


You can also use google gcloud tools to run the prediction.

    $ gcloud ml-engine local predict --model-dir=./linear/serve/ --json-instances linear/instances.json
