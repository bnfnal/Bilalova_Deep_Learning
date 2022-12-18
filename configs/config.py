# -*- coding: utf-8 -*-
"""Model config in json format"""
"""в конфигах мы определяем все, что можно настроить и изменить в будущем. 
Хорошими примерами являются гиперпараметры обучения, пути к папкам, архитектура модели, метрики, флаги.
Ваш эксперимент должен быть полностью описан здесь."""

CFG = {
    "datasets": {
        "data_path": "./configs/mnist/",
        "images_train_file_name": "train-images.idx3-ubyte",
        "labels_train_file_name": "train-labels.idx1-ubyte",
        "dataset_type": "train",
        "transforms": [],
        "nrof_classes": 10,
        "path": "MNIST",
        "image_size": 28,
        "load_with_info": "True",
        "shuffle": "True"
    },
    "train": {
        "batch_size": 32,
        "nrof_epoch": 20,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy", "balanced_accuracy"],
        "learning_rate": 0.001
    },
    "model": {
        "parametrs" : [('FullyConnected', {'input_size': 784, 'output_size': 128}), ('ReLU', {}), ('FullyConnected', {'input_size': 128, 'output_size': 10})]
    }
}

