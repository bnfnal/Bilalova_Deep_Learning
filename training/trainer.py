import numpy as np
from tensorboardX import SummaryWriter
from configs.config import CFG
from dataloader.dataloader import DataLoader
from datasets.mnist_dataset import MNISTDataset
from ops.normalize import Normalize
from ops.view import View
from enums.Datasets import DataSetType
from model.model import Model
from model.cross_entripy_loss import CrossEntripyLoss
from model.sgd import SGD

class Trainer:
    def __init__(self, cfg):
        self.net = Model(CFG['model']['parametrs'])
        self.loss = CrossEntripyLoss()
        self.optimiser = SGD(CFG['train']['learning_rate'], self.net, self.loss)
        self.epochs = CFG['train']['nrof_epoch']
        self.train_dataset = MNISTDataset(dataset_type=DataSetType.train,
                               transforms=[Normalize(), View()],
                               nrof_classes=cfg.data.nrof_classes)

        self.test_dataset = MNISTDataset(dataset_type=DataSetType.test,
                                         transforms=[Normalize(), View()],
                                         nrof_classes=cfg.data.nrof_classes,
                                         images_file='t10k-images.idx3-ubyte',
                                         labels_file='t10k-labels.idx1-ubyte'
                                         )

        self.train_dataloader = DataLoader(self.train_dataset,
                              cfg.train.batch_size,
                              None,
                              cfg.train.nrof_epoch)

        self.test_dataloader = DataLoader(self.test_dataset,
                                           cfg.train.batch_size,
                                           None,
                                           cfg.train.nrof_epoch)

        # next(self.train_dataloader.batch_generator())
        # self.train_dataloader.show_batch()

        self.writer = SummaryWriter("mlp")
        self.global_iteration = 0

    def fit(self):
        """
        функция обучения нейронной сети, включает в себя цикл по эпохам, в рамках одной эпохи:
        вызов train_epoch;
        вызов evaluate для обучающей и валидационной выборки последовательно
        """
        for epoch in range(self.epochs):
            self.train_epoch()
            self.evaluate(self.train_dataloader, 'train_loader')
            self.evaluate(self.test_dataloader, 'test_loader')

    def evaluate(self, dataloader: DataLoader, dataloader_type):
        """
        функция вычисления значения целевой функции и точности
        на всех данных из dataloader, сохранение результатов в tensorboard/mlflow/plotly
        :param dataloader: train_loader/valid_loader/test_loader
        """
        batch_loss_list = []  # значение целевой функции на каждой итерации
        batch_accuracy_list = []  # значение точности на каждой итерации
        batch_size_list = []   # размер батча на каждой итерации

        for batch, labels in dataloader.batch_generator():
            logits = self.net(batch)
            batch_loss = self.loss(logits, labels)
            batch_accuracy = (np.argmax(logits, 1) == labels).mean()
            batch_loss_list.append(batch_loss)
            batch_accuracy_list.append(batch_accuracy)
            batch_size_list.append(len(batch))

        total_loss = np.sum(np.dot(batch_loss_list, batch_size_list)) / np.sum(batch_size_list)
        total_accuracy = np.sum(np.dot(batch_accuracy_list, batch_size_list)) / np.sum(batch_size_list)
        self.writer.add_scalar(dataloader_type + "/total_loss", total_loss, self.global_iteration)
        self.writer.add_scalar(dataloader_type + "/total_accuracy", total_accuracy, self.global_iteration)


    def train_epoch(self):
        """
        функция обучения нейронной сети: для каждого батча из обучающей
        выборки вызывается метод _train_step и логируется значение целевой функции и точности на батче
        """
        self.net.train()
        for batch, labels in self.train_dataloader.batch_generator():
            self._train_step(batch, labels)
        self.net.eval()


    def _train_step(self, batch, labels):
        """
        один шаг обучения нейронной сети, в рамках которого происходит:
        1) вызов forward pass нейронной сети для батча;
        2) вычисление значения целевой функции;
        3) вызов функции minimize для вычисления обратного распространения и обновления весов
        """
        logits = self.net(batch)
        batch_loss = self.loss(logits, labels)
        batch_accuracy = (np.argmax(logits, 1) == labels).mean()
        self.optimiser.minimize()
        self.writer.add_scalar('batch_loss', batch_loss, self.global_iteration)
        self.writer.add_scalar('batch_accuracy',  batch_accuracy, self.global_iteration)
        print(self.net.phase, batch_loss, batch_accuracy)
        self.global_iteration += 1
        return batch_loss, batch_accuracy    # значения целевой функции и точности на этом батче

    def overfit_on_batch(self):
        batch, labels = next(self.train_dataloader.batch_generator())
        # берем 1 батч из выборки и на протяжении
        self.net.train()
        for i in range(1000):
            batch_loss, batch_accuracy = self._train_step(batch, labels)
            print(self.net.phase, batch_loss, batch_accuracy)
            # значение точности на протяжении 10 шагов подряд была 100%
        self.net.eval()

