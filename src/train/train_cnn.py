from dataclasses import dataclass
import glob
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data
from torch import optim
from sklearn.model_selection import StratifiedShuffleSplit

from src.preprocess.image_loader import ImgDataset, ImageTransform
# from src.preprocess.image_loader_self import DataLoaderSelf
from src.cnn.cnn import CNN

"""
GPU使用率が低い
・データ数が多いー＞違う
・画像サイズが大きい -> 違う

・
"""

@dataclass
class Config:
    lr: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.9
    input_dim: int = 600
    num_epoch: int = 100
    num_stopping: int = 2
    batch_size: int = 8
    save_path: str = '../../model/cnn.pt'


def train(train_dataloader, eval_dataloader, model, config):
    # check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Use device：", device)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)  # nn.CrossEntropyLoss = crass entropy + softmax, not have to one hot
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=[config.beta1, config.beta2])

    models = []
    eval_loss = []
    for epoch in range(config.num_epoch):
        t_epoch_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch, config.num_epoch))
        print('-------------')

        # ---------------------------------------------------------------------------------

        model.train()
        train_epoch_loss = 0
        for _images, _label in train_dataloader:
            # _images = _images.to(device)
            _images = _images.to(device).float()
            _label = _label.to(device)

            ########################################
            # _label = _label.to(device).long()
            ########################################

            pred = model(_images)

            loss = criterion(pred, _label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        print('Train epoch loss:{:.4f}'.format(train_epoch_loss / train_dataloader.batch_size))

        # -------------------------------------------------------------------------------

        model.eval()
        eval_epoch_loss = 0
        n_e = 0
        for _images, _label in eval_dataloader:
            # _images = _images.to(device)
            _images = _images.to(device).float()
            _label = _label.to(device)

            ########################################
            # _label = _label.to(device).long()
            ########################################

            pred = model(_images)

            loss = criterion(pred, _label)

            eval_epoch_loss += loss.item()
            n_e += 1

        models.append(model)

        # Early stopping -------------------------------------------------------

        eval_loss.append(eval_epoch_loss / n_e)

        if epoch >= config.num_stopping:
            if epoch == config.num_stopping:
                low_loss = np.min(eval_loss)
                low_index = np.argmin(eval_loss)
                if low_index == 0:
                    print('-------------------------------------------------------------------------------------------')
                    print("Early stopping")
                    print('Best Iteration:{}'.format(low_index + 1))
                    print('Best evaluation loss:{}'.format(low_loss))
                    break

            elif epoch == low_index + config.num_stopping:
                low_loss_new = np.min(eval_loss[low_index:])
                low_index_new = np.argmin(eval_loss[low_index:]) + low_index

                if low_loss <= low_loss_new:
                    print('-------------------------------------------------------------------------------------------')
                    print("Early stopping")
                    print('Best Iteration:{}'.format(low_index + 1))
                    print('Best evaluation loss:{}'.format(low_loss))
                    break
                else:
                    low_loss = low_loss_new
                    low_index = low_index_new
        else:
            pass

        t_epoch_finish = time.time()
        print('Eval_Epoch_Loss:{:.4f}'.format(eval_epoch_loss / n_e))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

        # check generated image ---------------------------------------------------
        # if epoch % 20 == 0:
        #     x = pred.to('cpu').detach().numpy().copy()
        #     x = x[0].reshape(128, 128)
        #     generated_images.append(x)
        #     plt.imshow(x)
        #     plt.show()

    return models[low_index + 1]


def process(image_dir_path, label_path, config):
    # read file path and label
    train_path_list = glob.glob(image_dir_path)
    train_path_list.sort()
    train_path_list = np.array(train_path_list)[:]

    label_df = pd.read_csv(label_path)
    label = label_df['label'].values[:]

    # split data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(train_path_list, label):
        x_train, x_test = train_path_list[train_index], train_path_list[test_index]
        y_train, y_test = label[train_index], label[test_index]

    for train_index, test_index in sss.split(x_train, y_train):
        x_train, x_eval = x_train[train_index], x_train[test_index]
        y_train, y_eval = y_train[train_index], y_train[test_index]

    # mk dataloader
    train_dataset = ImgDataset(file_list=x_train,
                               label_list=y_train,
                               transform=ImageTransform())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    eval_dataset = ImgDataset(file_list=x_eval,
                              label_list=y_eval,
                              transform=ImageTransform())
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=True)


    # train_dataloader = DataLoaderSelf(file_list=x_train,
    #                                   label_list=y_train,
    #                                   batch_size=config.batch_size,
    #                                   shuffle=True)
    #
    # eval_dataloader = DataLoaderSelf(file_list=x_eval,
    #                                  label_list=y_eval,
    #                                  batch_size=config.batch_size,
    #                                  shuffle=True)


    # model
    cnn = CNN(config.input_dim)

    # train model
    model, generated = train(train_dataloader, eval_dataloader, cnn, config)

    # save model
    torch.save(model.state_dict(), config.save_path)

    return generated


    #############################################################################################################
    # _mean = 0.5
    # _std = 0.5
    # train_path_list = glob.glob(rf'C:\Users\HirokiFuruyama\analysis\va_2021\figure\spectrogram_png\train\*.png')
    # label_t = np.zeros(len(train_path_list))
    # eval_path_list = glob.glob(rf'C:\Users\HirokiFuruyama\analysis\va_2021\figure\spectrogram_png\test\*.png')
    # label_e = np.zeros(len(eval_path_list))
    # train_dataset = ImgDataset(file_list=train_path_list,
    #                            label_list=label_t,
    #                            transform=ImageTransform(_mean, _std))
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    #
    # eval_dataset = ImgDataset(file_list=eval_path_list,
    #                           label_list=label_e,
    #                           transform=ImageTransform(_mean, _std))
    # eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=True)
    # cnn = CNN(config.input_dim, channels=1)
    # model, generated = train(train_dataloader, eval_dataloader, cnn, config)
    # torch.save(model.state_dict(), config.save_path)
    # return generated
    ###############################################################################################################


if __name__ == '__main__':
    train_dir = '../../data/train_images/*.jpg'
    label_dir = '../../data/train.csv'
    process(train_dir, label_dir, Config())
