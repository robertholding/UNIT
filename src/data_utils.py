#! /usr/bin/python3
# -*- coding:utf8 -*-

import pickle
import os
import glob

# from tensorflow.examples.tutorials.mnist import input_data
import scipy.io as sio
import cv2
import numpy as np

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.datasets import cifar10, cifar100, mnist, fashion_mnist
import matplotlib.image as img

class DataHandler:
    """
    this is a collection for the data processing methods for the model.
    """

    def __init__(self):
        pass

    # def mnist_tf(self, file_path):
        # """
        # mnist from tf
        # :return: tf tensor,
        # """
        # # read the original binary file
        # mnist = input_data.read_data_sets(file_path)

        # # mnist data
        # mnist_train = (mnist.train.images > 0). \
                          # reshape(55000, 28, 28, 1).astype('uint8') * 255
        # mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        # # print(mnist_train.shape)
        # mnist_test = (mnist.test.images > 0). \
                         # reshape(10000, 28, 28, 1).astype('uint8') * 255
        # mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
        # # print(mnist_test[0, :, :, 0])
        # mnist_valid = (mnist.validation.images > 0). \
                          # reshape(5000, 28, 28, 1).astype('uint8') * 255
        # mnist_valid = np.concatenate([mnist_valid, mnist_valid, mnist_valid], 3)

        # # mnist labels
        # mnist_train_label = mnist.train.labels
        # mnist_test_label = mnist.test.labels
        # mnist_valid_label = mnist.validation.labels

        # return (mnist_train, mnist_train_label), \
               # (mnist_test, mnist_test_label), \
               # (mnist_valid, mnist_valid_label)

    # def mnist_tf_xy(self, file_path):
        # """
        # mnist from tf, and then add additional normalization position,
        # x, y coordinate information  information for it.
        # :return: tf tensor,
        # """
        # # read the original binary file
        # mnist = input_data.read_data_sets(file_path)

        # # mnist data
        # mnist_train = (mnist.train.images > 0). \
                          # reshape(55000, 28, 28, 1).astype('uint8') * 255
        # mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        # mnist_train_xy = self.add_position_info(mnist_train)
        # print(mnist_train_xy.shape)

        # mnist_test = (mnist.test.images > 0). \
                         # reshape(10000, 28, 28, 1).astype('uint8') * 255
        # mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
        # mnist_test_xy = self.add_position_info(mnist_test)
        # # print(mnist_test[0, :, :, 0])

        # mnist_valid = (mnist.validation.images > 0). \
                          # reshape(5000, 28, 28, 1).astype('uint8') * 255
        # mnist_valid = np.concatenate([mnist_valid, mnist_valid, mnist_valid], 3)
        # mnist_valid_xy = self.add_position_info(mnist_valid)

        # # mnist labels
        # mnist_train_label = mnist.train.labels
        # mnist_test_label = mnist.test.labels
        # mnist_valid_label = mnist.validation.labels

        # return (mnist_train_xy, mnist_train_label), \
               # (mnist_test_xy, mnist_test_label), \
               # (mnist_valid_xy, mnist_valid_label)

    # def mnist_keras(self):
        # """
        # mnist from keras
        # :return: (x_train, y_train), (x_test, y_test)
        # """
        # return mnist.load_data()

    # def mnist_m(self, file_path):
        # """
        # read the mnist_m and the labels
        # :param file_path: where is the mnist_m.pkl
        # :return: (x_train, y_train), (x_test, y_test), (x_valid, y_valid)
        # """
        # # load and read the pickl file
        # mnist_m_path = os.path.join(file_path, 'mnistm_data.pkl')
        # print(mnist_m_path)
        # with open(mnist_m_path, 'rb') as fo:
            # # should use the with as, this method is as the same as SVHN
            # mnist_m = pickle.load(fo, encoding='bytes')

        # # get the mnist_m images data
        # mnist_m_train = mnist_m['train']
        # mnist_m_test = mnist_m['test']
        # mnist_m_valid = mnist_m['valid']

        # # get the mnist_m, its labels equal the mnist's
        # mnist = input_data.read_data_sets(file_path)
        # mnist_m_train_label = mnist.train.labels
        # mnist_m_test_label = mnist.test.labels
        # mnist_m_valid_label = mnist.validation.labels

        # return (mnist_m_train, mnist_m_train_label), \
               # (mnist_m_test, mnist_m_test_label), \
               # (mnist_m_valid, mnist_m_valid_label)

    # def mnist_m_multi(self, file_path):
        # """
        # read the mnist_multi with polluted background (i.e. mnistm) and the 
        # labels, it is created by creat_mnitm_multi.py first
        # :param file_path: where is the mnist_m.pkl
        # :return: (x_train, y_train), (x_test, y_test), (x_valid, y_valid)
        # """
        # # load and read the pickl file
        # mnist_m_path = os.path.join(file_path, 'mnistm_multi_data.pkl')
        # print(mnist_m_path)
        # with open(mnist_m_path, 'rb') as fo:
            # # should use the with as, this method is as the same as SVHN
            # mnist_m = pickle.load(fo, encoding='bytes')

        # # get the mnist_m images data
        # mnist_m_train = mnist_m['train']
        # mnist_m_test = mnist_m['test']
        # mnist_m_valid = mnist_m['valid']

        # # get the mnist_m, its labels equal the mnist's
        # mnist = input_data.read_data_sets(file_path)
        # mnist_m_train_label = mnist.train.labels
        # mnist_m_test_label = mnist.test.labels
        # mnist_m_valid_label = mnist.validation.labels

        # return (mnist_m_train, mnist_m_train_label), \
               # (mnist_m_test, mnist_m_test_label), \
               # (mnist_m_valid, mnist_m_valid_label)

    # def usps(self, file_path):
        # """
        # this is the data parser for usps.mat
        # :param file_path:
        # :return:
        # """
        # # load the mat file
        # file_path_data = os.path.join(file_path, "usps_all.mat")
        # usps = sio.loadmat(file_path_data)

        # # reshape and transpose the original data to one channel image format
        # data_usps = usps['data'].transpose(2, 1, 0). \
                                     # reshape(11000, 16, 16, 1). \
                                         # transpose(0, 2, 1, 3)
        # data_usps = np.concatenate([data_usps, data_usps, data_usps], 3)

        # # label
        # label_usps = np.zeros(10*1100)  # every category has 1100 samples
        # for i in range(10):
            # label_usps[1100*i:1100*(i + 1)] = np.tile((i + 1), 1100)
            # if i == 9:
                # label_usps[1100*i:1100*(i + 1)] = np.tile(0, 1100)

        # # shuffle
        # usps_data_n, usps_label_n = self.shuffle_aligned_list(
                                        # data=[data_usps, label_usps])

        # # get the train and test data, label
        # split_ratio = 0.8
        # split_s = 0
        # split_e = int(len(usps_data_n) * split_ratio)
        # len_data = len(data_usps)

        # data_train_x = usps_data_n[split_s:split_e]
        # label_train_y = usps_label_n[split_s:split_e]
        # data_test_x = usps_data_n[split_e:len_data]
        # label_test_y = usps_label_n[split_e:len_data]

        # return (data_train_x, label_train_y.astype(np.int32)), \
               # (data_test_x, label_test_y.astype(np.int32))

    def svhn_ufldl(self, file_path):
        """
        read the svhn dataset
        :param file_path: where svhn is
        :return: x_train, y_train, x_test, y_test
        """
        full_path_train = os.path.join(file_path, "train_32x32.mat")
        full_path_test = os.path.join(file_path, "test_32x32.mat")
        full_path_extra = os.path.join(file_path, "extra_32x32.mat")
        print(full_path_extra)

        ## train data
        data_train = sio.loadmat(full_path_train)
        x_train = data_train['X'].transpose(3, 0, 1, 2).astype('uint8')
        y_train_ovrall = data_train['y'].astype('uint8')

        # change the label format to np.array with shape(len(y_train),)
        y_train = np.ones((len(y_train_ovrall)))
        for i in range(len(y_train_ovrall)):
            y_train[i] = y_train_ovrall[i][0]
            if y_train[i] == 10:
                y_train[i] = 0

        ## test data
        data_test = sio.loadmat(full_path_test)
        x_test = data_test['X'].transpose(3, 0, 1, 2).astype('uint8')
        y_test_overall = data_test['y'].astype('uint8')

        # change the label format to np.array with shape(len(y_test),)
        y_test = np.ones((len(y_test_overall)))
        for i in range(len(y_test_overall)):
            y_test[i] = y_test_overall[i][0]
            if y_test[i] == 10:
                y_test[i] = 0

        ## extra data
        n_s = 0
        n_e = 70000
        split_ratio = 0.8
        split_s = 0
        split_e = int(n_e*split_ratio)

        # get all the extra data
        data_extra = sio.loadmat(full_path_extra)
        x_extra = data_extra['X'].transpose(3, 0, 1, 2).astype('uint8')
        y_extra = data_extra['y'].astype('uint8')
        # shuffle the data and label
        x_extra, y_extra = self.shuffle_aligned_list([x_extra, y_extra])

        x_extra_small = x_extra[n_s:n_e]  # get a small part 
        y_extra_small = y_extra[n_s:n_e]

        del x_extra

        x_extra_train = x_extra_small[split_s:split_e]
        y_extra_train_1 = y_extra_small[split_s:split_e].astype('uint8')

        x_extra_test = x_extra_small[split_e:n_e]
        y_extra_test_1 = y_extra_small[split_e:n_e].astype('uint8')

        # change the label format to np.array with shape(len(y_test),)
        y_extra_train = np.ones((len(y_extra_train_1)))
        for i in range(len(y_extra_train_1)):
            y_extra_train[i] = y_extra_train_1[i][0]
            if y_extra_train[i] == 10:
                y_extra_train[i] = 0

        y_extra_test = np.ones((len(y_extra_test_1)))
        for i in range(len(y_extra_test_1)):
            y_extra_test[i] = y_extra_test_1[i][0]
            if y_extra_test[i] == 10:
                y_extra_test[i] = 0



        return (x_train, y_train.astype(np.int32)), \
               (x_test, y_test.astype(np.int32)), \
               (x_extra_train, y_extra_train.astype(np.int32)), \
               (x_extra_test, y_extra_test.astype(np.int32))

    # def svhn_ufldl_xy(self, file_path):
        # """
        # read the svhn dataset
        # :param file_path: where svhn is
        # :return: x_train, y_train, x_test, y_test
        # """
        # full_path_train = os.path.join(file_path, "train_32x32.mat")
        # full_path_test = os.path.join(file_path, "test_32x32.mat")
        # full_path_extra = os.path.join(file_path, "extra_32x32.mat")

        # # train data
        # data_train = sio.loadmat(full_path_train)
        # x_train = data_train['X'].transpose(3, 0, 1, 2).astype('uint8')
        # y_train_ovrall = data_train['y'].astype('uint8')

        # # train data: add the x and y coordinate information, as two 
        # # additional channels of the original image data
        # x_train_xy = self.add_position_info(x_train)

        # ## change the label format to np.array with shape(len(y_train),)
        # y_train = np.ones((len(y_train_ovrall)))
        # for i in range(len(y_train_ovrall)):
            # y_train[i] = y_train_ovrall[i][0]
            # if y_train[i] == 10:
                # y_train[i] = 0

        # # test data
        # data_test = sio.loadmat(full_path_test)
        # x_test = data_test['X'].transpose(3, 0, 1, 2).astype('uint8')
        # y_test_overall = data_test['y'].astype('uint8')

        # # test data: add the x and y coordinate information, as two additional 
        # # channels of the original image data
        # x_test_xy = self.add_position_info(x_test)

        # # change the label format to np.array with shape(len(y_test),)
        # y_test = np.ones((len(y_test_overall)))
        # for i in range(len(y_test_overall)):
            # y_test[i] = y_test_overall[i][0]
            # if y_test[i] == 10:
                # y_test[i] = 0

        # ## extra data
        # n_s = 0
        # n_e = 70000
        # split_ratio = 0.8
        # split_s = 0
        # split_e = int(n_e*split_ratio)

        # data_extra = sio.loadmat(full_path_extra)
        # x_extra = data_extra['X'].transpose(3, 0, 1, 2).astype('uint8')
        # y_extra = data_extra['y'].astype('uint8')
        # # shuffle the data and label
        # x_extra, y_extra = self.shuffle_aligned_list([x_extra, y_extra])

        # x_extra_small = x_extra[n_s:n_e]  # get a small part 
        # y_extra_small = y_extra[n_s:n_e]

        # del x_extra

        # x_extra_train = x_extra_small[split_s:split_e]
        # y_extra_train_1 = y_extra_small[split_s:split_e].astype('uint8')

        # x_extra_test = x_extra_small[split_e:n_e]
        # y_extra_test_1 = y_extra_small[split_e:n_e].astype('uint8')

        # # add position information for extra small dataset
        # x_extra_train_xy = self.add_position_info(data_img=x_extra_train)
        # x_extra_test_xy = self.add_position_info(data_img=x_extra_test)

        # # change the label format to np.array with shape(len(y_test),)
        # y_extra_train_xy = np.ones((len(y_extra_train_1)))
        # for i in range(len(y_extra_train_1)):
            # y_extra_train_xy[i] = y_extra_train_1[i][0]
            # if y_extra_train_xy[i] == 10:
                # y_extra_train_xy[i] = 0

        # y_extra_test_xy = np.ones((len(y_extra_test_1)))
        # for i in range(len(y_extra_test_1)):
            # y_extra_test_xy[i] = y_extra_test_1[i][0]
            # if y_extra_test_xy[i] == 10:
                # y_extra_test_xy[i] = 0

        # return (x_train_xy, y_train.astype(np.int32)), \
               # (x_test_xy, y_test.astype(np.int32)), \
               # (x_extra_train_xy, y_extra_train_xy.astype(np.int32)), \
               # (x_extra_test_xy, y_extra_test_xy.astype(np.int32))


    # def cifar10_alex(self, file_path):
        # """
        # read the cifar10 from original data, and to parse to one dataset
        # :param file_path: where cifar10 is
        # :return: x_train, y_train, x_test, y_test
        # """
        # # train data setting
        # data_all_train = np.zeros((50000, 3072)).astype('uint8')
        # label_all_train = np.zeros(50000, ).astype('uint8')

        # # # test data setting
        # # data_all_test = np.zeros((10000, 3072))
        # # label_all_test = np.zeros(10000,)

        # # global setting
        # batch_size = 10000

        # full_path_train = os.path.join(file_path, 'data*')
        # full_path_test = os.path.join(file_path, 'test_batch')
        # # glob.glob find all the related file
        # print(sorted(glob.glob(full_path_train)))

        # # to get all the train data and correspond labels
        # for batch, index in zip(sorted(glob.glob(full_path_train)), range(5)):
            # print(batch, index)
            # with open(batch, 'rb') as fo:
                # data_batch_train = pickle.load(fo, encoding='bytes')
                # #     dict_all[os.path.split(batch)[1]] = data_batch
                # data_all_train[index * batch_size:index * batch_size
                               # + batch_size] = \
                                   # data_batch_train[b'data'].astype('uint8')
                # # list to np.array
                # label_batch = np.array(data_batch_train[b'labels'])
                # label_all_train[index * batch_size:index * batch_size
                                # + batch_size] = label_batch.astype('uint8')

        # with open(full_path_test, 'rb') as fo:
            # data_batch_test = pickle.load(fo, encoding='bytes')
            # data_all_test = data_batch_test[b'data'].astype('uint8')
            # label_all_test = np.array(data_batch_test[b'labels']).astype('uint8')
        # # first 1024, then second 1024
        # data_all_train = \
            # data_all_train.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1)
        # data_all_test = data_all_test.reshape((10000, 3, 32, 32)). \
                                          # transpose(0, 2, 3, 1)

        # return (data_all_train, label_all_train), \
               # (data_all_test, label_all_test)

    # def cifar100_alex(self, file_path):
        # """
        # read the cifar100 data from the original dataset
        # :param file_path: where the cifar100 is
        # :return: x_train, y_train, x_test, y_test
        # """
        # full_path_train = os.path.join(file_path, 'train')
        # full_path_test = os.path.join(file_path, 'test')

        # with open(full_path_train, 'rb') as fo:
            # data_batch_train = pickle.load(fo, encoding='bytes')
            # data_all_train = data_batch_train[b'data'].astype('uint8')
            # coarse_label_all_train \
                    # = np.array(
                          # data_batch_train[b'coarse_labels']).astype('uint8')
            # fine_label_all_train \
                    # = np.array(data_batch_train[b'fine_labels']).astype('uint8')

        # with open(full_path_test, 'rb') as fo:
            # data_batch_test = pickle.load(fo, encoding='bytes')
            # data_all_test = data_batch_test[b'data'].astype('uint8')
            # coarse_label_all_test \
                    # = np.array(data_batch_test[b'coarse_labels']).astype('uint8')
            # fine_label_all_test \
                    # = np.array(data_batch_test[b'fine_labels']).astype('uint8')

        # data_all_train = data_all_train.reshape((50000, 3, 32, 32)). \
                                            # transpose(0, 2, 3, 1)
        # data_all_test = data_all_test.reshape((10000, 3, 32, 32)). \
                                          # transpose(0, 2, 3, 1)

        # return (data_all_train, fine_label_all_train, coarse_label_all_train), \
               # (data_all_test, fine_label_all_test, coarse_label_all_test)

    # def cifar10_keras(self):
        # """
        # read cifar10 from keras datasets
        # :return: x_train, y_train, x_test, y_test
        # """
        # return cifar10.load_data()

    # def cifar100_keras(self, lable_model="fine"):
        # """
        # read cifar100 from keras datasets
        # :param lable_model:
        # :return: x_train, y_train(fine or coarse), x_test, y_test(fine or coarse)
        # """
        # return cifar100.load_data(label_mode=lable_model)

    def office31(self, file_path, pre_processing=False, flag_conc=False):
        """
        read the raw image data from office 31 datasets, then return the
        images np array, and their labels
        """
        # get all the file path

        files_name = []
        for dir_info in os.walk(file_path):
            for file_name in dir_info[2]:
                file_name_one = os.path.join(dir_info[0], file_name)
                files_name.append(file_name_one)

        files_name_sorted = sorted(files_name)
        # print(files_name_sorted)

        class_name = []
        file_path_images = os.path.join(file_path, "images/*")
        for dir_info in glob.glob(file_path_images):
            class_name_one = os.path.split(dir_info)[-1]
            class_name.append(class_name_one)

        class_name_sorted = sorted(class_name)
        # print(class_name_sorted)

        # read and put the image into one np array
        if pre_processing:
            img_width = 90
            img_height = 90
            data_all \
                    = np.ones((len(files_name_sorted),
                               img_width, img_height)).astype('uint8')
            for (i, file_path) in enumerate(files_name_sorted):
                img = cv2.imread(file_path)
                # img_reshape = img.reshape(1, img.shape[0], img.shape[1], 3)
                img_gray = self.gray(data_image=img)
                img_resize = self.resize(img_gray, (img_width, img_height))
                data_all[i, :, :] = img_resize
            if flag_conc:
                data_all = data_all.reshape(-1, img_width, img_height, 1)
                data_all = np.concatenate([data_all, data_all, data_all], 3)
        else:
            img_width = 28
            img_height = 28
            img_channle = 3
            data_all = np.ones((len(files_name_sorted), img_width,
                                img_height, img_channle)).astype('uint8')
            for (i,file_path) in enumerate(files_name_sorted):
                # print(i, j)
                img = cv2.imread(file_path)
                img_reshape = img.reshape(1, img.shape[0], img.shape[1], 3)
                img_resize = self.resize(img_reshape, (img_width, img_height))
                data_all[i, :, :, :] = img_resize

        label_all = np.ones(len(files_name_sorted))

        for (f_index, f_name) in enumerate(files_name_sorted):
            for (c_index, c_name) in enumerate(class_name_sorted):
                if c_name in f_name:
                    label_all[f_index] = c_index

        # get the train and test data, label
        split_ratio = 0.8
        split_s = 0
        split_e = int(len(data_all) * split_ratio)
        len_data = len(data_all)

        # random the data and its label
        data_all, label_all = self.shuffle_aligned_list([data_all, label_all])

        data_train_x = data_all[split_s:split_e]
        label_train_y = label_all[split_s:split_e]
        data_test_x = data_all[split_e:len_data]
        label_test_y = label_all[split_e:len_data]


        return (data_train_x, label_train_y.astype(np.int32)), \
               (data_test_x, label_test_y.astype(np.int32))

    def office31_surf(self, file_path, pre_processing=False, flag_conc=False):
        """
        read the raw image data from office 31 datasets, then return the
        images np array, and their labels
        """
        # get all the file path

        files_name = []
        for dir_info in os.walk(file_path):
            for file_name in dir_info[2]:
                file_name_one = os.path.join(dir_info[0], file_name)
                files_name.append(file_name_one)

        files_name_sorted = sorted(files_name)
        print(files_name_sorted[0:10])

        class_name = []
        file_path_images = os.path.join(file_path, "interest_points/*")
        for dir_info in glob.glob(file_path_images):
            class_name_one = os.path.split(dir_info)[-1]
            class_name.append(class_name_one)

        class_name_sorted = sorted(class_name)
        print(class_name_sorted[:])
        print(class_name)

        # read and put the image into one np array
        data_all = np.ones((len(files_name_sorted), 800))

        for i, file_path in enumerate(files_name_sorted):
            # print(sio.loadmat(file_path))
            if 'dslr' in file_path:
                if 'amazon' in file_path:
                    data_all[i, :] = sio.loadmat(file_path)['histogram']
            else:
                data_all[i, :] = sio.loadmat(file_path)['histogram']

        if pre_processing:
            img_width = 28
            img_height = 28
            data_all = data_all[:, 0:784]
            data_all = data_all.reshape(-1, img_width, img_height, 1)
            if flag_conc:
                data_all = np.concatenate([data_all, data_all, data_all], 3)

        label_all = np.ones(len(files_name_sorted))

        for (f_index, f_name) in enumerate(files_name_sorted):
            for (c_index, c_name) in enumerate(class_name_sorted):
                if c_name in f_name:
                    label_all[f_index] = c_index

        # get the train and test data, label
        split_ratio = 0.8
        split_s = 0
        split_e = int(len(data_all) * split_ratio)
        len_data = len(data_all)

        # random the data and its label
        data_all, label_all = self.shuffle_aligned_list([data_all, label_all])

        data_train_x = data_all[split_s:split_e]
        label_train_y = label_all[split_s:split_e]
        data_test_x = data_all[split_e:len_data]
        label_test_y = label_all[split_e:len_data]


        return (data_train_x, label_train_y.astype(np.int32)), \
               (data_test_x, label_test_y.astype(np.int32))

    def imagenet(self, file_path):
        """
        this is mini imagenet dataset
        """
        data_all = np.zeros((1281167, 3072)).astype('uint8')
        label_all = np.zeros(1281167, ).astype('uint8')

        # global setting
        batch_size = 128116

        full_path = os.path.join(file_path, 'train_data_batch_*')
        # glob.glob find all the related file
        print(sorted(glob.glob(full_path)))

        # to get all the train data and correspond labels
        for batch, index in zip(sorted(glob.glob(full_path)), range(10)):
            print(batch, index)
            with open(batch, 'rb') as fo:
                data_batch = pickle.load(fo, encoding='bytes')
                label_batch = np.array(data_batch['labels'])  # list to np.array
                if index < 9:
                    data_all[index * batch_size:index * batch_size
                             + batch_size] = data_batch['data'].astype('uint8')
                    label_all[index * batch_size:index * batch_size
                              + batch_size] = label_batch.astype('uint8')
                else:
                    data_all[batch_size*9:1281167, :] \
                            = data_batch['data'].astype('uint8')
                    label_all[batch_size*9:1281167] \
                            = label_batch.astype('uint8')
        # first 1024, then second 1024
        data_all = \
            data_all.reshape((1281167, 3, 32, 32)).transpose(0, 2, 3, 1)

        # get the train and test data, label
        split_ratio = 0.8
        split_s = 0
        split_e = int(len(data_all) * split_ratio)
        len_data = len(data_all)

        # random the data and its label
        data_all, label_all = self.shuffle_aligned_list([data_all, label_all])

        data_train_x = data_all[split_s:split_e]
        label_train_y = label_all[split_s:split_e]
        data_test_x = data_all[split_e:len_data]
        label_test_y = label_all[split_e:len_data]


        return (data_all_train, label_all_train), \
               (data_all_test, label_all_test)

    def imagenet_tiny(self,
                      file_path,
                      img_width,
                      img_height):
        """
        this will return the tiny imagenet, 200 categories, every category 500
        images for test, 20 for val, 50 for test.
        """
        # get all the file path

        files_name = []
        for dir_info in os.walk(file_path):
            for file_name in dir_info[2]:
                if file_name.endswith('JPEG'):
                    file_name_one = os.path.join(dir_info[0], file_name)
                    files_name.append(file_name_one)

        files_name_sorted = sorted(files_name)
        # print(files_name_sorted)

        class_name = []
        file_path_images = os.path.join(file_path, "*")
        # print(file_path_images)
        for dir_info in glob.glob(file_path_images):
            class_name_one = os.path.split(dir_info)[-1]
            class_name.append(class_name_one)

        class_name_sorted = sorted(class_name)

        #print(class_name_sorted)

        img_channle = 3

        data_len = len(files_name_sorted)

        data_all = np.ones((data_len, img_width,
                            img_height, img_channle)).astype('uint8')
        label_all = np.ones(data_len,)

        for (i, img_path_one) in enumerate(files_name_sorted):
            img = cv2.imread(img_path_one).astype('uint8')
            img_resize = self.resize(img, (img_width, img_height))
            data_all[i][:, :, :] = img_resize

        for (img_index, img_path_one) in enumerate(files_name_sorted):
            for (lable_index, label_name) in enumerate(class_name_sorted):
                if label_name in img_path_one:
                    label_all[img_index] = lable_index

        split_ratio = 0.8
        split_s = 0
        split_e = int(len(data_all) * split_ratio)
        len_data = len(data_all)

        # random the data and its label
        data_all, label_all = self.shuffle_aligned_list([data_all, label_all])

        data_train_x = data_all[split_s:split_e]
        label_train_y = label_all[split_s:split_e]
        data_test_x = data_all[split_e:len_data]
        label_test_y = label_all[split_e:len_data]


        return (data_train_x, label_train_y), (data_test_x, label_test_y)

    def linemod(self, file_path):
        """
        this is the linemod parser, to return the image and dpth, the synth
        and tmpl as train dataset, and real as test
        """
        # get all the file name and the label name
        files_name = []
        for dir_info in os.walk(file_path):
            for file_name in dir_info[2]:
                file_name_one = os.path.join(dir_info[0], file_name)
                files_name.append(file_name_one)

        files_name_sorted = sorted(files_name)
        # print(files_name_sorted)

        class_name = []
        file_path_images = os.path.join(file_path, "*")
        # print(file_path_images)
        for dir_info in glob.glob(file_path_images):
            class_name_one = os.path.split(dir_info)[-1]
            class_name.append(class_name_one)

        class_name_sorted = sorted(class_name)

        # get all the image files and the dpt files name
        img_train_files_sorted = []
        img_test_files_sorted = []
        dpt_train_files_sorted = []
        dpt_test_files_sorted = []

        for file_name in files_name_sorted:
            if "img" in file_name:
                if "synth" in file_name or "tmpl" in file_name:
                    img_train_files_sorted.append(file_name)
                else:
                    img_test_files_sorted.append(file_name)
            else:
                if "synth" in file_name or "tmpl" in file_name:
                    dpt_train_files_sorted.append(file_name)
                else:
                    dpt_test_files_sorted.append(file_name)
        print(img_train_files_sorted[0:2])
        print(dpt_train_files_sorted[0:2])
        # images and labels for image
        img_train_x = np.ones((len(img_train_files_sorted),
                               64, 64, 3)).astype('uint8')
        img_test_x = np.ones((len(img_test_files_sorted),
                              64, 64, 3)).astype('uint8')
        img_train_y = np.ones(len(img_train_files_sorted),)
        img_test_y = np.ones(len(img_test_x),)

        for (i, file_name) in enumerate(img_train_files_sorted):
            img_one = cv2.imread(file_name)
            # img_train_x[i] = img_one
            img_train_x[i] = cv2.cvtColor(img_one, cv2.COLOR_BGR2RGB)
        for (i, file_name) in enumerate(img_test_files_sorted):
            img_one = cv2.imread(file_name)
            # img_test_x[i] = img_one
            img_test_x[i] = cv2.cvtColor(img_one, cv2.COLOR_BGR2RGB)

        for (img_index, img_path_one) in enumerate(img_train_files_sorted):
            for (lable_index, label_name) in enumerate(class_name_sorted):
                if label_name in img_path_one:
                    img_train_y[img_index] = lable_index

        for (img_index, img_path_one) in enumerate(img_test_files_sorted):
            for (lable_index, label_name) in enumerate(class_name_sorted):
                if label_name in img_path_one:
                    img_test_y[img_index] = lable_index

        # images and labels for dpt
        dpt_train_x = np.ones((len(dpt_train_files_sorted),
                               64, 64, 3)).astype('uint8')
        dpt_test_x = np.ones((len(dpt_test_files_sorted),
                              64, 64, 3)).astype('uint8')
        dpt_train_y = np.ones(len(dpt_train_files_sorted), )
        dpt_test_y = np.ones(len(dpt_test_files_sorted), )

        for (i, file_name) in enumerate(dpt_train_files_sorted):
            img_one = cv2.imread(file_name)
            dpt_train_x[i] = cv2.cvtColor(img_one, cv2.COLOR_BGR2RGB)
        for (i, file_name) in enumerate(dpt_test_files_sorted):
            img_one = cv2.imread(file_name)
            dpt_test_x[i] = cv2.cvtColor(img_one, cv2.COLOR_BGR2RGB)

        for (img_index, img_path_one) in enumerate(dpt_train_files_sorted):
            for (lable_index, label_name) in enumerate(class_name_sorted):
                if label_name in img_path_one:
                    dpt_train_y[img_index] = lable_index

        for (img_index, img_path_one) in enumerate(dpt_test_files_sorted):
            for (lable_index, label_name) in enumerate(class_name_sorted):
                if label_name in img_path_one:
                    dpt_test_y[img_index] = lable_index

        (img_train_x_1, img_train_y_1), (img_train_x_2, img_train_y_2) \
                = self.split_data(img_train_x, img_train_y)
        (img_test_x_1, img_test_y_1), (img_test_x_2, img_test_y_2) \
                = self.split_data(img_test_x, img_test_y)
        (dpt_train_x_1, dpt_train_y_1), (dpt_train_x_2, dpt_train_y_2) \
                = self.split_data(dpt_train_x, dpt_train_y)
        (dpt_test_x_1, dpt_test_y_1), (dpt_test_x_2, dpt_test_y_2) \
                = self.split_data(dpt_test_x, dpt_test_y)


        return (img_train_x_1, img_train_y_1.astype(np.int32)), \
               (img_train_x_2, img_train_y_2.astype(np.int32)), \
               (img_test_x_1, img_test_y_1.astype(np.int32)), \
               (img_test_x_2, img_test_y_2.astype(np.int32)), \
               (dpt_train_x_1, dpt_train_y_1.astype(np.int32)), \
               (dpt_train_x_2, dpt_train_y_2.astype(np.int32)), \
               (dpt_test_x_1, dpt_test_y_1.astype(np.int32)), \
               (dpt_test_x_2, dpt_test_y_2.astype(np.int32))

    def linemod_dpt(self, file_path):
        """
        this is the linemod parser, to return the image and dpth, the synth
        and tmpl as train dataset, and real as test
        """
        # get all the file name and the label name
        files_name = []
        for dir_info in os.walk(file_path):
            for file_name in dir_info[2]:
                file_name_one = os.path.join(dir_info[0], file_name)
                files_name.append(file_name_one)

        files_name_sorted = sorted(files_name)
        # print(files_name_sorted)

        class_name = []
        file_path_images = os.path.join(file_path, "*")
        # print(file_path_images)
        for dir_info in glob.glob(file_path_images):
            class_name_one = os.path.split(dir_info)[-1]
            class_name.append(class_name_one)

        class_name_sorted = sorted(class_name)

        # get all the image files and the dpt files name
        img_train_files_sorted = []
        img_test_files_sorted = []
        dpt_train_files_sorted = []
        dpt_test_files_sorted = []

        for file_name in files_name_sorted:
            if "img" in file_name:
                if "synth" in file_name or "tmpl" in file_name:
                    img_train_files_sorted.append(file_name)
                else:
                    img_test_files_sorted.append(file_name)
            else:
                if "synth" in file_name or "tmpl" in file_name:
                    dpt_train_files_sorted.append(file_name)
                else:
                    dpt_test_files_sorted.append(file_name)

        # images and labels for image
        img_train_x = np.ones((len(img_train_files_sorted),
                               64, 64, 6)).astype('uint8')
        img_test_x = np.ones((len(img_test_files_sorted),
                              64, 64, 6)).astype('uint8')
        img_train_y = np.ones(len(img_train_files_sorted),)
        img_test_y = np.ones(len(img_test_x),)

        for (i, img_name, dpt_name) in zip(range(len(img_train_x)),
                                           img_train_files_sorted,
                                           dpt_train_files_sorted):
            img_one = cv2.imread(img_name)
            dpt_one = cv2.imread(dpt_name)
            # img_train_x[i] = img_one
            img_train_x[i,:,:,0:3] = cv2.cvtColor(img_one, cv2.COLOR_BGR2RGB)
            img_train_x[i,:,:,3:6] = cv2.cvtColor(dpt_one, cv2.COLOR_BGR2RGB)

        for (i, img_name, dpt_name) in zip(range(len(img_test_x)),
                                           img_test_files_sorted,
                                           dpt_test_files_sorted):
            img_one = cv2.imread(img_name)
            dpt_one = cv2.imread(dpt_name)
            # img_test_x[i] = img_one
            img_test_x[i,:,:,0:3] = cv2.cvtColor(img_one, cv2.COLOR_BGR2RGB)
            img_test_x[i,:,:,3:6] = cv2.cvtColor(dpt_one, cv2.COLOR_BGR2RGB)

        for (img_index, img_path_one) in enumerate(img_train_files_sorted):
            for (lable_index, label_name) in enumerate(class_name_sorted):
                if label_name in img_path_one:
                    img_train_y[img_index] = lable_index

        for (img_index, img_path_one) in enumerate(img_test_files_sorted):
            for (lable_index, label_name) in enumerate(class_name_sorted):
                if label_name in img_path_one:
                    img_test_y[img_index] = lable_index


        (img_train_x_1, img_train_y_1), (img_train_x_2, img_train_y_2) \
                = self.split_data(img_train_x, img_train_y)
        (img_test_x_1, img_test_y_1), (img_test_x_2, img_test_y_2) \
                = self.split_data(img_test_x, img_test_y)


        return (img_train_x_1, img_train_y_1.astype(np.int32)), \
               (img_train_x_2, img_train_y_2.astype(np.int32)), \
               (img_test_x_1, img_test_y_1.astype(np.int32)), \
               (img_test_x_2, img_test_y_2.astype(np.int32))

    def caltech101_li(self):
        pass

    def caltech256_li(self):
        pass

    # def fashion_mnist_keras(self):
        # """
        # read the fashion dataset like mnist from keras
        # :return:x_train, y_train, x_test, y_test
        # """
        # (fashion_train_x, fashion_train_y), (fashion_test_x, fashion_test_y) \
                # = fashion_mnist.load_data()

        # fashion_train_x = fashion_train_x.reshape(len(fashion_train_x),
                                                  # 28, 28, 1)
        # fashion_test_x = fashion_test_x.reshape(len(fashion_test_x), 28, 28, 1)

        # fashion_train_x = np.concatenate([fashion_train_x, fashion_train_x,
                                          # fashion_train_x], 3)
        # fashion_test_x = np.concatenate([fashion_test_x, fashion_test_x,
                                         # fashion_test_x], 3)

        # return (fashion_train_x, fashion_train_y), \
               # (fashion_test_x, fashion_test_y)

    # def fashion_m(self, file_path):
        # """
        # read the fashion_m and the labels, it is created by the
        # creat_fashion_m.py first.
        # :param file_path: where is the mnist_m.pkl
        # :return: (x_train, y_train), (x_test, y_test), (x_valid, y_valid)
        # """
        # # load and read the pickl file
        # fashion_m_path = os.path.join(file_path, 'fashionm_data.pkl')
        # print(fashion_m_path)
        # with open(fashion_m_path, 'rb') as fo:
            # # should use the with as, this method is as the same as SVHN
            # fashion_m = pickle.load(fo, encoding='bytes')

        # # get the mnist_m images data
        # fashion_m_train = fashion_m['train']
        # fashion_m_test = fashion_m['test']

        # # get the mnist_m, its labels equal the mnist's
        # (fashion_train_x, fashion_train_y), (fashion_test_x, fashion_test_y) \
                # = self.fashion_mnist_keras()
        # fashion_m_train_label = fashion_train_y
        # fashion_m_test_label = fashion_test_y

        # return (fashion_m_train, fashion_m_train_label), \
               # (fashion_m_test, fashion_m_test_label)


    def resize(self, data_image, res_shape):
        """
        resize data to the shape
        :param data_image: which one wants to be resized
        :param res_shape: the shape to be resized
        :return: rescaled data
        """
        if len(data_image.shape) == 4:
            data_resized = np.zeros((len(data_image), res_shape[0],
                                     res_shape[1],
                                     data_image.shape[3])).astype('uint8')

            for i in range(len(data_image)):
                data_resized[i] = cv2.resize(data_image[i], res_shape,
                                             interpolation
                                             =cv2.INTER_NEAREST).astype('uint8')

        elif len(data_image.shape) == 3:
            if data_image.shape[2] == 3:  # a single RGB image
                data_resized = np.zeros((res_shape[0], res_shape[1],
                                         data_image.shape[2])).astype('uint8')
                data_resized = cv2.resize(data_image, res_shape,
                                          interpolation
                                          =cv2.INTER_NEAREST).astype('uint8')
            else:  # batch of gray image
                data_resized = np.zeros((len(data_image), res_shape[0],
                                         res_shape[1])).astype('uint8')
                for i in range(len(data_image)):
                    data_resized[i] \
                            = cv2.resize(data_image[i], res_shape,
                                         interpolation
                                         =cv2.INTER_NEAREST).astype('uint8')

        else:  # a single gray image
            data_resized = np.zeros((res_shape[0], res_shape[1]))
            data_resized = cv2.resize(data_image, res_shape,
                                      interpolation
                                      =cv2.INTER_NEAREST).astype('uint8')

        return data_resized

    def gray(self, data_image):
        """
        read image data and convert to gray
        :param data_image: image data
        :return: gray image
        """
        if len(data_image.shape) == 4:
            # print(type(data_image))
            data_gray = np.zeros((data_image.shape[0], data_image.shape[1],
                                  data_image.shape[2])).astype('uint8')
            # print(data_gray.shape)
            for i in range(len(data_image)):
                # convert to gray image
                data_gray[i] = cv2.cvtColor(data_image[i],
                                            cv2.COLOR_BGR2GRAY).astype('uint8')
        else:
            data_gray = np.zeros((data_image.shape[0],
                                  data_image.shape[1])).astype('uint8')
            data_gray =  cv2.cvtColor(data_image,
                                      cv2.COLOR_BGR2GRAY).astype('uint8')

        return data_gray

    def bw(self, data_image):
        """
        read the image data to convert to binary
        :param data_image: gray image data
        :return: black and white image
        """
        print(type(data_image), data_image.shape[1], data_image.shape[2])
        data_bw = np.zeros((len(data_image), data_image.shape[1],
                            data_image.shape[2])).astype('uint8')

        for i in range(len(data_image)):
            # use the adaptive method to binary
            data_bw[i] = cv2.adaptiveThreshold(data_image[i], 1,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY,
                                               11, 2).astype('uint8')

            # covert to black and white use simple method
            # thresh, data_bw[i] = cv2.threshold(data_image[i], 128, 1,
            #                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        return data_bw

    def resize_gray_bw(self, data_image, res_shape):
        """
        read image data and convert to bw
        :param data_image: data
        :param res_shape: resize shape
        :return:
        """
        data_bw = np.zeros((len(data_image), res_shape[0],
                            res_shape[1])).astype('uint8')

        if (data_image.shape[1], data_image.shape[2]) == res_shape:
            for i in range(len(data_image)):
                data_gray = cv2.cvtColor(data_image[i],
                                         cv2.COLOR_BGR2GRAY).astype('uint8')
                data_bw[i] = cv2.adaptiveThreshold(data_gray, 1,
                                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY,
                                                   11, 2).astype('uint8')
        else:
            for i in range(len(data_image)):
                data_resize = cv2.resize(data_image[i], res_shape,
                                         interpolation
                                         =cv2.INTER_CUBIC).astype('uint8')
                data_gray = cv2.cvtColor(data_resize,
                                         cv2.COLOR_BGR2GRAY).astype('uint8')
                data_bw[i] = cv2.adaptiveThreshold(data_gray, 1,
                                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY,
                                                   11, 2).astype('uint8')

        return data_bw

    def remove_category(self, data_image, labels_image, category=4):
        """
        read the image data and remove a specific category
        :param data_image: image data
        :param labels_image: image label
        :param category: a specific category, 4 is deer and 7 is horse in the 
         cifar10
        :return: data_remove,
        """

        # to remove a specific category
        index_y = np.where(labels_image == category)  # remove the deer first
        print("There are %d samples in category %d" % (len(index_y[0]),
                                                       category))

        data_remove = np.delete(data_image, index_y[0], 0)
        labels_remove = np.delete(labels_image, index_y[0], 0)

        if len(data_image.shape) == 4:  # for RGB
            data_specific_category = np.zeros((len(index_y[0]),
                                              data_image.shape[1],
                                              data_image.shape[2],
                                              data_image.shape[3]))
        elif len(data_image.shape) == 3:  # for Gray
            data_specific_category = np.zeros((len(index_y[0]),
                                              data_image.shape[1],
                                              data_image.shape[2]))
        elif len(data_image.shape) == 2:  # for reshape image
            data_specific_category = np.zeros((len(index_y[0]),
                                              data_image.shape[1]))
        else:
            raise ValueError("Error in Dimension")

        for i, j in zip(range(len(index_y[0])), index_y[0]):
            data_specific_category[i] = data_image[j]

        print(data_remove)
        return (data_remove, labels_remove), data_specific_category

    def batch_next(self, data_image, size_batch):
        pass

    def shuffle_aligned_list(self, data):
        """Shuffle arrays in a list by shuffling each array identically."""
        num = data[0].shape[0]
        p = np.random.permutation(num)
        return [d[p] for d in data]

    def batch_generator(self, data, size_batch, shuffle=True):
        """Generate batches of data.

        Given a list of array-like objects, generate batches of a given
        size by yielding a list of array-like objects corresponding to the
        same slice of each input.
        """
        if shuffle:
            data = self.shuffle_aligned_list(data)

        batch_count = 0
        while True:
            if batch_count * size_batch + size_batch >= len(data[0]):
                batch_count = 0

                if shuffle:
                    data = self.shuffle_aligned_list(data)

            start = batch_count * size_batch
            end = start + size_batch
            print(start, end)
            batch_count += 1
            yield [d[start:end] for d in data]

    def add_position_info(self, data_img):
        """
        this function to add the additional position information for the
        image data_img
        Arg:
            data_img: will be add x and y position information for it,
            should be (samples, width, height, channels)
        """
        # add additional position information channel to the mnist digits
        # shape 
        len_data = data_img.shape[0]
        width_data = data_img.shape[1]
        height_data = data_img.shape[2]

        # build three empty np array to store the new data
        x_position = np.ones((len_data, width_data,
                              height_data, 1)).astype('uint8')
        y_position = np.ones((len_data, width_data,
                              height_data, 1)).astype('uint8')

        # build the normalization position information 
        for i in range(width_data):
            for j in range(height_data):
                x_position[:][i][j] = int(j * (255 / width_data))
                y_position[:][j][i] = int(j * (255 / width_data))

        # get the 5-channel image data
        data_img_xy = np.concatenate((data_img, x_position, y_position), axis=3)

        return data_img_xy.astype('uint8')

    def batch_augment_generator(self,
                                data,
                                label,
                                brightness_range=None,
                                size_batch=32):
        """
        this will augment the data with keras.preprocessing.image.
        ImageDataGenerator
        """
        datagen = ImageDataGenerator(featurewise_center=False,
                                     featurewise_std_normalization=False,
                                     rotation_range=20,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.1,
                                     brightness_range=brightness_range,
                                     horizontal_flip=False,
                                     vertical_flip= False)

        datagen.fit(data)

        data_g = datagen.flow(data, label, batch_size=size_batch)

        batch_count = 0
        while True:
            # print(batch_count)
            data_next, label_next = next(data_g)
            # batch_count*size_batch + size_batch < len(data):
            if  data_next.shape[0] < size_batch:
                data_g = datagen.flow(data, label, batch_size=size_batch)
                data_next, label_next = next(data_g)
            #print(data_next.shape)
            batch_count += 1
            start = batch_count * size_batch
            end = start + size_batch
            print(start, end)
            yield (data_next, label_next)

    def mnist_multi_digit(self, dataset):
        """
        this method add another digits to the original one, to format a
        similar SVHN ones
        """

        width_mix = 40
        height_mix = 40

        width_mix_s = 30
        height_mix_s = 40

        width_mix_s1 = 20
        height_mix_s1 = 40

        length_data = len(dataset)

        data_mix = np.ones((length_data, 28, 28, 3))

        data_new = np.ones((width_mix, height_mix, 3))
        data_new_small = np.ones((width_mix_s, height_mix_s, 3))
        data_new_small_1 = np.ones((width_mix_s1, height_mix_s1, 3))

        pos_list_x = [(0, 20), (10, 30), (20, 40)]
        pos_list_y = [(0, 11), (29, 40)]

        pos_list_2 = [(0, 20, 0, 11, 20, 40, 29, 40),
                      (10, 30, 0, 11, 10, 30, 29, 40),
                      (20, 40, 0, 11, 0, 20, 29, 40)]

        for i in range(length_data):

            mix_type = np.random.choice(3, 1)
            # print(mix_type)

            if mix_type[0] == 0:
                # data_new[10:30, 10:30, :] = mnist_train_x[i][4:24, 4:24, :]
                # data_mix[i] = Data.resize(data_image=data_new, res_shape=(28, 28))
                data_mix[i] = dataset[i]

            elif mix_type[0] == 1:
                data_new[10:32, 10:30, :] = dataset[i][4:26, 4:24, :]

                mix_x = np.random.choice(3, 1)
                mix_y = np.random.choice(2, 1)

                x_s = pos_list_x[mix_x[0]][0]
                x_e = pos_list_x[mix_x[0]][1]
                y_s = pos_list_y[mix_y[0]][0]
                y_e = pos_list_y[mix_y[0]][1]

                mix_digit = np.random.choice(length_data, 1)
                # print(mix_x, mix_y, mix_digit)
                # print("1 digit:", i)
                data_new[x_s:x_e, y_s:y_e, :] \
                        = dataset[mix_digit[0]][4:24, 5:16, :]

                if x_s == 0:
                    data_new_small = data_new[0:30, :, :]
                    data_mix[i] = self.resize(data_image=data_new_small,
                                              res_shape=(28, 28))
                elif x_s == 10:
                    data_new_small_1 = data_new[10:30, :, :]
                    data_mix[i] = self.resize(data_image=data_new_small_1,
                                              res_shape=(28, 28))
                elif x_s == 20:
                    data_new_small = data_new[10:40, :, :]
                    data_mix[i] = self.resize(data_image=data_new_small,
                                              res_shape=(28, 28))

                # data_mix[i] = self.resize(data_image=data_new, res_shape=(28, 28))

            elif mix_type == 2:
                data_new[10:32, 10:30, :] = dataset[i][4:26, 4:24, :]
                mix_p = np.random.choice(3, 1)
                mix_digit = np.random.choice(length_data, 2)
                # print(mix_p, mix_digit)

                x_s_1 = pos_list_2[mix_p[0]][0]
                x_e_1 = pos_list_2[mix_p[0]][1]
                y_s_1 = pos_list_2[mix_p[0]][2]
                y_e_1 = pos_list_2[mix_p[0]][3]

                x_s_2 = pos_list_2[mix_p[0]][4]
                x_e_2 = pos_list_2[mix_p[0]][5]
                y_s_2 = pos_list_2[mix_p[0]][6]
                y_e_2 = pos_list_2[mix_p[0]][7]


                data_new[x_s_1:x_e_1, y_s_1:y_e_1, :] \
                        = dataset[mix_digit[0]][4:24, 5:16, :]
                data_new[x_s_2:x_e_2, y_s_2:y_e_2, :] \
                        = dataset[mix_digit[1]][4:24, 5:16, :]

                if x_s_1 == 10:
                    data_new_small_1 = data_new[10:30, :, :]
                    data_mix[i] = self.resize(data_image=data_new_small_1,
                                              res_shape=(28, 28))
                else:
                    data_mix[i] = self.resize(data_image=data_new,
                                              res_shape=(28, 28))
            data_new = np.ones((width_mix, height_mix, 3))
            data_new_small = np.ones((width_mix_s, height_mix_s, 3))
            data_new_small_1 = np.ones((width_mix_s1, height_mix_s1, 3))

        return data_mix

    def split_data(self, data, label):
        """
        this splits the data and label to train and test
        """
        # split the data
        split_ratio = 0.8
        split_s = 0
        split_e = int(len(data) * split_ratio)
        len_data = len(data)

        # random the data and its label
        img_train_x, img_train_y = self.shuffle_aligned_list([data, label])

        data_train_x = img_train_x[split_s:split_e]
        label_train_y = img_train_y[split_s:split_e]
        data_test_x = img_train_x[split_e:len_data]
        label_test_y = img_train_y[split_e:len_data]


        return (data_train_x, label_train_y), (data_test_x, label_test_y)
