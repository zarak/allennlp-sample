// For more info on config files generally, see https://guide.allennlp.org/using-config-files
{
    "dataset_reader" : {
        // This name needs to match the name that you used to register your dataset reader, with
        // the call to `@DatasetReader.register()`.
        "type": "classification-tsv",
        // These other parameters exactly match the constructor parameters of your dataset reader class.
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },
    "train_data_path": "data/train.text.trimmed.tsv",
    "validation_data_path": "data/test.text.trimmed.tsv",
    "model": {
        // This name needs to match the name that you used to register your model, with
        // the call to `@Model.register()`.
        "type": "simple_classifier",
        // These other parameters exactly match the constructor parameters of your model class.
        "embedder": {
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",
                    "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
                    "do_layer_norm": false,
                    "dropout": 0.0
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 256,
            "hidden_size": 50,
            "num_layers": 1,
            "bidirectional": true
        }
    },
    "data_loader": {
        // See http://docs.allennlp.org/master/api/data/dataloader/ for more info on acceptable
        // parameters here.
        "batch_size": 2,
        "shuffle": true
    },
    "trainer": {
        // See http://docs.allennlp.org/master/api/training/trainer/#gradientdescenttrainer-objects
        // for more info on acceptable parameters here.
        "optimizer": "adam",
        "num_epochs": 5,
        "patience": 20
    }
    // There are a few other optional parameters that can go at the top level, e.g., to configure
    // vocabulary behavior, to use a separate dataset reader for validation data, or other things.
    // See http://docs.allennlp.org/master/api/commands/train/#from_partial_objects for more info
    // on acceptable parameters.
}
