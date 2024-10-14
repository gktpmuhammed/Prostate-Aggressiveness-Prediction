import yaml

config_file = "config/config.yaml"
cross_val_config_file = "config/cross_val_config.yaml"

def load_config():
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        validate_config(config)
        return config

def validate_config(config):

    model_name = config['model']
    if model_name not in ["MEDICALNET", "RADIMAGENET_1_SLICE", "RADIMAGENET_3_SLICES", "THREE_SLICE_CNN", "DENSENET_3D", "SIMPLE_CNN_3D"]:
        raise ValueError("Invalid model name: {}".format(model_name))

    dataset_name = config["dataset"]
    if dataset_name not in ["PRIVATE_1_SLICE", "PRIVATE_ONLY_TUMOR", "PICAI_3_SLICE_TUMOR", "PRIVATE_PROSTATE", "PICAI_1_SLICE", "PICAI_ONLY_TUMOR"]:
        raise ValueError("Invalid dataset name: {}".format(dataset_name))

    compatible_models = {
        "PRIVATE_1_SLICE": ["RADIMAGENET_1_SLICE"],
        "PRIVATE_PROSTATE": ["MEDICALNET"],
        "PRIVATE_ONLY_TUMOR": ["MEDICALNET", "DENSENET_3D", "SIMPLE_CNN_3D"],
        "PICAI_ONLY_TUMOR": ["MEDICALNET", "DENSENET_3D"],
        "PICAI_3_SLICE": ["RADIMAGENET_3_SLICES", "THREE_SLICE_CNN"],
        "PICAI_1_SLICE": ["RADIMAGENET_1_SLICE"],
    }

    if model_name not in compatible_models[dataset_name]:
        raise ValueError(f"Invaid model {model_name} for dataset {datset_name}")

    train_portion = config["train_portion"]
    if not (0 < train_portion <= 1):
        raise ValueError("Train portion must be between 0 and 1, found: {}".format(train_portion))

    num_classes = config["num_classes"]
    if not (num_classes > 1 and isinstance(num_classes, int)):
        raise ValueError("Number of classes must be an integer between 1 and 5, found: {}".format(num_classes))

    lr = config['hparams']['lr']
    if not (0 < lr <= 1):
        raise ValueError("Learning rate must be between 0 and 1, found: {}".format(lr))

    batch_size = config['hparams']['batch_size']
    if not (batch_size > 0 and isinstance(batch_size, int)):
        raise ValueError("Batch size must be a positive integer, found: {}".format(batch_size))

    weight_decay = config['hparams']['weight_decay']
    if not (0 <= weight_decay <= 1):
        raise ValueError("Weight decay must be between 0 and 1, found: {}".format(weight_decay))

    epochs = config['hparams']['epochs']
    if not (epochs > 0 and isinstance(epochs, int)):
        raise ValueError("Epochs must be a positive integer, found: {}".format(epochs))

    train_pretrained_weight = config['hparams']['train_pretrained_weight']
    if not (0 <= train_pretrained_weight <= 1):
        raise ValueError("train_pretrained_weight must be between 0 and 1, found: {}".format(train_pretrained_weight))

    if dataset_name in ["PICAI_ONLY_TUMOR", "PICAI_1_SLICE", "PICAI_3_SLICE_TUMOR"] and num_classes != 3:
        raise ValueError(f"Num Classes must be 3 when model {dataset_name} is used!")

    print("Configuration is valid.")