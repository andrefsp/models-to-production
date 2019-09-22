## Linear model

To train the model simply run the python file.
    
    $ python train_task.py --train-file ./data/linear.train.csv  --export-path ./serve/

or using files and storing on gcloud

    $ python train_task.py --train-file gs://ai-learning/models/linear/data/linear.train.csv  --export-path gs://ai-learning/models/serve/


It will save the model on a `./serve/` folder.

You can than run predictions using `saved_model_cli`

    $ saved_model_cli run --dir ./serve/ --tag_set serve --signature_def serving_default --input_exprs='X=np.float32(4)'


You can also use google gcloud tools to run the prediction.

    $ gcloud ml-engine local predict --model-dir=./serve/ --json-instances instances.json
