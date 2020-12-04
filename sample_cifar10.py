import random
import numpy as np
import random
import os
import calculate_kl_with_other
import argparse
from tensorflow import keras

def sample_uniform(args):
    expname = args.expname
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train = x_train.reshape((-1, 32, 32, 3)) / 255.0
    x_test = x_test.reshape((-1, 32, 32, 3)) / 255.0

    shuffle_ix = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_ix]
    y_train = y_train[shuffle_ix]

    dataset_id = args.id
    n = 5000

    for i in range(args.punum):
        np.save(expname+'/' + str(dataset_id) + '/x_train_' + str(i) + '.npy', x_train[int(i * n):int((i + 1) * n)])
        np.save(expname+'/' + str(dataset_id) + '/y_train_' + str(i) + '.npy', y_train[int(i * n):int((i + 1) * n)])


def sample_random(args):
    expname = args.expname
    dataset = []
    for j in range(10):
        dataset_path = "cif/split_by_categorical/x_train_category" + str(j) + ".npy"
        dataset.append(np.load(dataset_path))
    prnum = args.pnum - args.punum
    for k in range(prnum):  # k users_num a[i] each class percentage
        # random each class percentage
        a = []
        for i in range(10):
            a.append(random.randint(1, 100))
        a = a / np.sum(a)

        for i in range(10):
            random_index = random.sample(range(1, 5000), int(a[i].tolist() * 5000))

            if (i == 0):
                out_x_train = dataset[i][random_index]
                out_y_train = np.array([[i]] * int(a[i].tolist() * 5000))
            else:
                out_tmp_x_train = dataset[i][random_index]
                out_x_train = np.concatenate((out_x_train, out_tmp_x_train))
                out_tmp_y_train = np.array([[i]] * int(a[i].tolist() * 5000))
                out_y_train = np.concatenate((out_y_train, out_tmp_y_train))
        out_y_train = keras.utils.to_categorical(out_y_train, 10)
        np.save(expname + '/' + str(args.id) + '/x_train_' + str(args.pnum-1-k) + '.npy', out_x_train)
        np.save(expname + '/' + str(args.id) + '/y_train_' + str(args.pnum-1-k) + '.npy', out_y_train)

        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-pnum', type=int, default=5, help='number of clients')
    parser.add_argument('-id', type=int, default=1, help='dataset_id')
    parser.add_argument('-punum', type=int, default=4, help='number of uniform clients')
    parser.add_argument('-expname', type=str, default='different_u_r', help='number of random clients')
    args = parser.parse_args()
    expname = args.expname
    if not os.path.exists(expname + "/" + str(args.id)):
        os.makedirs(expname + "/" + str(args.id))
    sample_uniform(args)
    sample_random(args)

    calculate_kl_with_other.kl_with_uniform(args)