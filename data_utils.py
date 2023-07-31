import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
from collections import defaultdict
import config


def load_data():
    '''
    load training,validation,testing data
    :return: train_data, user_ratings, test_data_np, test, vali_data, vali, user_no, item_no, train_matrix
    '''
    # define a dictionary to store the train data's user-item relation
    user_ratings = defaultdict(set)

    # load Amazon training data's user-item relation
    with open(config.train_data_path, 'r', encoding='utf-8-sig') as f:
        for line in f.readlines():
            u, i = line.split(",")
            u = int(u)
            i = int(i)
            user_ratings[u].add(i)
    # load Amazon training data
    train_data = pd.read_csv(config.train_data_path, sep=',', header=None, names=['user', 'item'], usecols=[0, 1],
                             dtype={0: np.int32, 1: np.int32})
    # get unique user and item number for Amazon data
    user_no = train_data['user'].max() + 1
    item_no = train_data['item'].max() + 1
    train_data = train_data.values.tolist()


    # #  load MovieLens100K training data's user-item relation
    # with open(config.train_data_path, 'r') as f:
    #     for line in f.readlines():
    #         u, i = line.split(" ")
    #         u = int(u)-1
    #         i = int(i)-1
    #         user_ratings[u].add(i)
    #     # load MovieLens100K training data
    # train_data = pd.read_csv(config.train_data_path, sep=' ', header=None, names=['user', 'item'], usecols=[0, 1],
    #                                  dtype={0: np.int32, 1: np.int32})
    #
    # train_data = train_data.values.tolist()
    # # define unique user and item number for MovieLens100K data
    # user_no = 943
    # item_no = 1682

    # transformed training samples into sparse matrices
    train_matrix = sp.dok_matrix((user_no, item_no), dtype=np.float32)

    # preprocess the data for MovieLens100K, since the MovieLens100K data index start from 1

    # for x in train_data:
    #     x[0]-=1
    #     x[1]-=1

    # set the positive item for the training data matrix
    for x in train_data:
        train_matrix[x[0], x[1]] = 1.0

    size_u_i = user_no * item_no
    test_data_np = np.zeros((user_no, item_no))
    vali_data = np.zeros((user_no, item_no))
    test = np.zeros(size_u_i)
    vali = np.zeros(size_u_i)

    # load Amazon testing data
    with open(config.test_data_path, 'r', encoding='utf-8-sig') as f:
        line = f.readline()
        while line is not None and line != '':
            arr = line.split(',')
            u = int(arr[0])
            i = int(arr[1])
            test_data_np[u][i] = 1
            line = f.readline()

    # # load MovieLens100K testing data
    # file = open(config.test_data_path, 'r')
    # for line in file:
    #     line = line.split(' ')
    #
    #     user = int(line[0])-1
    #     item = int(line[1])-1
    #     test_data_np[user][item] = 1

    # load Amazon validation data
    with open(config.vali_data_path, 'r', encoding='utf-8-sig') as f:
        line = f.readline()
        while line is not None and line != '':
            arr = line.split(',')
            u = int(arr[0])
            i = int(arr[1])
            vali_data[u][i] = 1
            line = f.readline()
    # flatten the testing matrix
    for u in range(user_no):
        for item in range(item_no):
            if int(test_data_np[u][item]) == 1:
                test[u * item_no + item] = 1
            else:
                test[u * item_no + item] = 0
    # flatten the validation matrix
    for u in range(user_no):
        for item in range(item_no):
            if int(vali_data[u][item]) == 1:
                vali[u * item_no + item] = 1
            else:
                vali[u * item_no + item] = 0

    return train_data, user_ratings, test_data_np, test, vali_data, vali, user_no, item_no, train_matrix
    # below return for MovieLens100K data
    # return train_data, user_ratings, test_data_np, test, user_no, item_no, train_matrix


class BPRData(data.Dataset):
    def __init__(self, features,
                 item_no, train_matrix=None, neg_no=0, is_training=None):
        super(BPRData, self).__init__()
        self.features = features
        self.item_no = item_no
        self.train_matrix = train_matrix
        self.neg_no = neg_no
        self.is_training = is_training
        # print(item_no)

    def ng_sample(self):
        '''
        sample negative items when training
        :return:
        '''
        assert self.is_training, 'judge if it is training phase'
        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            for t in range(self.neg_no):
                j = np.random.randint(self.item_no)
                while (u, j) in self.train_matrix:
                    j = np.random.randint(self.item_no)
                self.features_fill.append([u, i, j])

    def __len__(self):
        return self.neg_no * len(self.features) if \
            self.is_training else len(self.features)

    def __getitem__(self, idx):
        features = self.features_fill if \
            self.is_training else self.features

        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if \
            self.is_training else features[idx][1]

        return user, item_i, item_j
