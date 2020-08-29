# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils import data
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math
import time
import os
import random
import csv
from visdom import Visdom
import collections
import shutil


class GenerateImageData:
    def __init__(self):
        self.data_path = ['data/I/', 'data/II/']  # 数据位置
        self.expand_ratio = 0.25  # 人脸框扩张率
        self.trainset_ratio = 0.8  # 训练集占比
        self.boundary_size = 112  # 裁剪图片尺寸
        self.shuffle_seed = 42  # 数据集洗牌随机种子
        self.neg_per_pos = 3  # 负样本生成次数
        self.data_save_path = ['stage3_train_data/', 'stage3_test_data/']  # 数据位置
        if args.re_generate_data:  # 若已生成过图片集，是否再次生成图片集
            self.data_dic = self.read_txt()  # 调用读取txt函数，得到数据字典
            self.train_set, self.test_set = self.split_train_test()  # 分割训练验证集
            self.generate_data()  # 生成图片集

    def read_txt(self):  # 读取2个label.txt文件中的数据，返回以字典形式的存储
        data_dic = []
        for sub_path in self.data_path:
            label_data = open(sub_path + 'label.txt')
            label_lines = label_data.readlines()
            label_data.close()
            for single_line in label_lines:
                single_line_split = single_line.split()
                img_path = sub_path + single_line_split[0]
                num_data = [float(_i) for _i in single_line_split[1:]]
                if min(num_data) > 0:  # 去掉存在负值的样本
                    frame_corner = [int(num_data[0]), int(num_data[1]), int(num_data[2]), int(num_data[3])]
                    landmarks = np.array(num_data[4:])  # 类型为np.array
                    data_dic.append({'img_path': img_path, 'frame_corner': frame_corner, 'landmarks': landmarks})
        return data_dic

    def split_train_test(self):  # 分割训练集和验证集
        np.random.seed(self.shuffle_seed)  # 输入随机种子，保证训练集、验证集隔离
        shuffled_data = np.random.permutation(self.data_dic)
        train_size = int(len(self.data_dic) * self.trainset_ratio)  # 根据训练集占比得到训练集大小
        return shuffled_data[:train_size], shuffled_data[train_size:]

    def generate_data(self):  # 返回单个样本的数据和标签，供DataLoader调用
        for _f in self.data_save_path:  # make directory
            if os.path.exists(_f):  # 若已存在，先删除文件夹
                shutil.rmtree(_f)
            time.sleep(3)  # 为完全删除文件夹留3s缓冲时间
            os.makedirs(_f)
        
        data_set = [self.train_set, self.test_set]
        for _s in range(2):
            file = open(self.data_save_path[_s] + 'label.txt', 'w')  # 生成训练集
            name_list = []
            for _i in data_set[_s]:
                name = _i['img_path'].split("/", 2)[-1]
                name_list.append(name)
                name_count = collections.Counter(name_list)
                name_times = name_count[name]
                name_split = name.split('.')
                img_save_path = self.data_save_path[_s] + name_split[0] + '_' + str(name_times) + '.' + name_split[1]
    
                img_crop, relative_landmarks = self.expand_crop(_i)  # 扩大人脸框并剪裁
                img, label = self.proportion_resize(img_crop, label=relative_landmarks)  # 保持x y轴比例进行resize
                path_label = img_save_path + ' ' + '1'
                for _j in label:
                    path_label = path_label + ' ' + str(_j)
                file.write(path_label + chr(10))
                cv2.imwrite(img_save_path, img)
                # 生成负样本
                img_ori = cv2.imread(_i['img_path'], cv2.IMREAD_COLOR)  # 读入
                h, w, _ = img_ori.shape
                x1, y1, x2, y2 = _i['frame_corner']
                w_expand = (x2 - x1) * self.expand_ratio
                h_expand = (y2 - y1) * self.expand_ratio
                e_x1 = int(max(0, x1 - w_expand))
                e_y1 = int(max(0, y1 - h_expand))
                e_x2 = int(min(w - 1, x2 + w_expand))
                e_y2 = int(min(h - 1, y2 + h_expand))
                e_frame_w = e_x2 - e_x1
                e_frame_h = e_y2 - e_y1
                neg_x1_max = w - 1 - e_frame_w
                neg_y1_max = h - 1 - e_frame_h
                for _k in range(self.neg_per_pos):  # 每个正样本形成self.neg_per_pos个负样本
                    _t = 0
                    while _t < 30:  # 在生成每个负样本时，执行30次尝试
                        a1 = random.randint(0, neg_x1_max)
                        a2 = a1 + e_frame_w
                        b1 = random.randint(0, neg_y1_max)
                        b2 = b1 + e_frame_h
                        iou_is_ok = True
                        for _c in data_set[_s]:
                            if _c['img_path'] == _i['img_path']:
                                roi_x1, roi_y1, roi_x2, roi_y2 = _c['frame_corner']
                                width_expand = (roi_x2 - roi_x1) * self.expand_ratio
                                height_expand = (roi_y2 - roi_y1) * self.expand_ratio
                                e_roi_x1 = int(max(0, roi_x1 - width_expand))
                                e_roi_y1 = int(max(0, roi_y1 - height_expand))
                                e_roi_x2 = int(min(w - 1, roi_x2 + width_expand))
                                e_roi_y2 = int(min(h - 1, roi_y2 + height_expand))
                                iou = self.cal_iou(e_roi_x1, e_roi_y1, e_roi_x2, e_roi_y2, a1, b1, a2, b2)
                                if iou > 0.3:  # 如果iou超过0.3，重新生成随机框
                                    iou_is_ok = False
                                    _t += 1
                                    break
                        if iou_is_ok:  # 与同一图片中的所有人脸框iou < 0.3，则存储图片并跳出while循环
                            name_list.append(name)
                            name_count = collections.Counter(name_list)
                            name_times = name_count[name]
                            name_split = name.split('.')
                            img_save_path = self.data_save_path[_s] + name_split[0] + '_' + str(name_times) + '.' + \
                                            name_split[1]
                            neg_img_crop = img_ori[b1:b2, a1:a2, :]
                            img_save, _ = self.proportion_resize(neg_img_crop)  # 保持x y轴比例进行resize
                            file.write(img_save_path + ' ' + '0' + chr(10))
                            cv2.imwrite(img_save_path, img_save)
                            break
            file.close()

        # file = open(self.data_save_path[1] + 'label.txt', 'w')  # 生成测试集
        # name_list = []
        # for _i in self.test_set:
        #     name = _i['img_path'].split("/", 2)[-1]
        #     name_list.append(name)
        #     name_count = collections.Counter(name_list)
        #     name_times = name_count[name]
        #     name_split = name.split('.')
        #     img_save_path = self.data_save_path[1] + name_split[0] + '_' + str(name_times) + '.' + name_split[1]
        # 
        #     img_crop, relative_landmarks = self.expand_crop(_i)  # 扩大人脸框并剪裁
        #     img, label = self.proportion_resize(img_crop, label=relative_landmarks)  # 保持x y轴比例进行resize
        #     path_label = img_save_path + ' ' + '1'
        #     for _j in label:
        #         path_label = path_label + ' ' + str(_j)
        #     file.write(path_label + chr(10))
        #     cv2.imwrite(img_save_path, img)
        #     # 生成负样本
        #     x1, y1, x2, y2 = _i['frame_corner']
        #     img = cv2.imread(_i['img_path'], cv2.IMREAD_COLOR)  # 读入
        #     h, w, _ = img.shape
        #     new_box_x1_max = int(w - (x2 - x1) * (1 + 2 * self.expand_ratio))
        #     new_box_y1_max = int(h - (y2 - y1) * (1 + 2 * self.expand_ratio))
        #     for _k in range(3):  # 每个正样本形成3个负样本
        #         while _t < 30:  # 在生成每个负样本时，执行30次尝试
        #             a1 = random.randint(0, new_box_x1_max)
        #             a2 = a1 + w
        #             b1 = random.randint(0, new_box_y1_max)
        #             b2 = b1 + h
        #             iou_is_ok = True
        #             for _c in self.train_set:
        #                 if _c['img_path'] == _i['img_path']:
        #                     roi_x1, roi_y1, roi_x2, roi_y2 = _c['frame_corner']
        #                     width_expand = (roi_x2 - roi_x1) * self.expand_ratio
        #                     height_expand = (roi_y2 - roi_y1) * self.expand_ratio
        #                     e_roi_x1 = int(max(0, roi_x1 - width_expand))
        #                     e_roi_y1 = int(max(0, roi_y1 - height_expand))
        #                     e_roi_x2 = int(min(width - 1, roi_x2 + width_expand))
        #                     e_roi_y2 = int(min(height - 1, roi_y2 + height_expand))
        #                     iou = self.cal_iou(e_roi_x1, e_roi_y1, e_roi_x2, e_roi_y2, a1, b1, a2, b2)
        #                     if iou > 0.3:  # 如果iou超过0.3，重新生成随机框
        #                         iou_is_ok = False
        #                         break
        #             if iou_is_ok:  # 与同一图片中的所有人脸框iou < 0.3，则存储图片并跳出while循环
        #                 name_list.append(name)
        #                 name_count = collections.Counter(name_list)
        #                 name_times = name_count[name]
        #                 name_split = name.split('.')
        #                 img_save_path = self.data_save_path[1] + name_split[0] + '_' + str(name_times) + '.' + \
        #                                 name_split[1]
        #                 img_crop = img[a1:a2, b1:b2]
        #                 img, _ = self.proportion_resize(img_crop)  # 保持x y轴比例进行resize
        #                 file.write(img_save_path + ' ' + '0' + chr(10))
        #                 cv2.imwrite(img_save_path, img)
        #                 break
        # file.close()

        # file = open(self.data_save_path[1] + 'label.txt', 'w')  # 生成测试集
        # name_list = []
        # for _i in self.test_set:
        #     name = _i['img_path'].split("/", 2)[-1]
        #     name_list.append(name)
        #     name_count = collections.Counter(name_list)
        #     name_times = name_count[name]
        #     name_split = name.split('.')
        #     img_save_path = self.data_save_path[1] + name_split[0] + '_' + str(name_times) + '.' + name_split[1]
        # 
        #     img_crop, relative_landmarks = self.expand_crop(_i)  # 扩大人脸框并剪裁
        #     img, label = self.proportion_resize(img_crop, relative_landmarks)  # 保持x y轴比例进行resize
        #     str_path_label = img_save_path
        #     for _j in label:
        #         str_path_label = str_path_label + ' ' + str(_j)
        #     file.write(str_path_label + chr(10))
        #     cv2.imwrite(img_save_path, img)
        # file.close()

    def expand_crop(self, single_face_data):  # 对单张图片扩大人脸框，返回剪裁后结果及相对剪裁框左上角的标记点坐标
        img = cv2.imread(single_face_data['img_path'], cv2.IMREAD_COLOR)  # 灰度方式读入
        height, width, _ = img.shape
        roi_x1, roi_y1, roi_x2, roi_y2 = single_face_data['frame_corner']
        # 计算水平方向和垂直方向扩张尺寸
        width_expand, height_expand = (roi_x2 - roi_x1) * self.expand_ratio, (roi_y2 - roi_y1) * self.expand_ratio
        e_roi_x1 = int(max(0, roi_x1 - width_expand))
        e_roi_y1 = int(max(0, roi_y1 - height_expand))
        e_roi_x2 = int(min(width - 1, roi_x2 + width_expand))
        e_roi_y2 = int(min(height - 1, roi_y2 + height_expand))
        e_roi_x1_y1 = np.array([e_roi_x1, e_roi_y1])
        relative_landmarks = single_face_data['landmarks'].reshape(-1, 2) - e_roi_x1_y1  # 计算相对于左上角的坐标
        img_crop = img[e_roi_y1:e_roi_y2, e_roi_x1:e_roi_x2]
        return img_crop, relative_landmarks

    def proportion_resize(self, img, label=None):  # 保持x y轴比例进行resize
        height, width, _ = img.shape
        ratio = self.boundary_size / max(height, width)  # 得到缩放比
        new_img = cv2.resize(np.asarray(img), (round(ratio * width), round(ratio * height)))
        new_h, new_w, _ = new_img.shape
        # 使用保持x y比例方式resize，当宽高不相等时，进行填充
        top, bottom, left, right = 0, 0, 0, 0
        if new_w == new_h:  # 对宽高相等的情况，图片填充为零
            pass
        elif new_w == self.boundary_size:  # 若缩放后w与缩放尺寸相同，则填充h
            need_pad = self.boundary_size - new_h
            top, bottom = int(need_pad / 2), math.ceil(need_pad / 2)
        else:  # 若缩放后h与缩放尺寸相同，则填充w
            need_pad = self.boundary_size - new_w
            left, right = int(need_pad / 2), math.ceil(need_pad / 2)
        if label is not None:
            resize_label = label * ratio + np.array([left, top])  # 计算填充后关键点位置
            label = resize_label.flatten()  # 对关键点进行压平操作
        return cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_REPLICATE), label

    @staticmethod
    def cal_iou(x1, y1, x2, y2, a1, b1, a2, b2):
        left = max(x1, a1)
        right = min(x2, a2)
        top = max(y1, b1)
        bottom = min(y2, b2)
        if left >= right or top >= bottom:
            iou = 0
        else:
            overlap = (right - left) * (bottom - top)
            iou = overlap / ((x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - overlap)
        return iou


class DataSet(data.Dataset):
    def __init__(self, transform=None):
        generate_data = GenerateImageData()
        self.data_save_path = generate_data.data_save_path  # 数据位置
        self.transform = transform  # 输入数据转换
        train_set, test_set = self.read_txt()  # 调用读取txt函数，得到数据字典
        print(len(train_set))
        print(len(test_set))
        if args.re_cal_mean_std:
            self.data_mean, self.data_std = self.cal_mean_std(train_set)
            print(f'data_mean: {self.data_mean}  data_std: {self.data_std}')
        else:
            self.data_mean, self.data_std = [0.382, 0.424, 0.525], [0.256, 0.260, 0.284]  # use cv2 BGR
        self.data_set = train_set if args.work in ['train', 'finetune'] else test_set  # 选定数据集

    def read_txt(self):  # 读取2个label.txt文件中的数据，返回以字典形式的存储
        train_test = []
        for sub_path in self.data_save_path:
            sub_data = []
            label_data = open(sub_path + 'label.txt')
            label_lines = label_data.readlines()
            label_data.close()
            for single_line in label_lines:
                single_line_split = single_line.split()
                img_path = single_line_split[0]
                is_face = single_line_split[1]
                landmarks = np.array([float(_i) for _i in single_line_split[1:]]) if is_face == 1 else np.array([])
                sub_data.append({'img_path': img_path, 'is_face': is_face, 'landmarks': landmarks})
            train_test.append(sub_data)
        return train_test

    @staticmethod
    def cal_mean_std(train_set):
        img = cv2.imread(train_set[0]['img_path'], cv2.IMREAD_COLOR)
        imgs = img[:, :, :, np.newaxis]
        for _i in range(1, len(train_set)):
            if _i % 500 == 0:
                print(_i)
            img = cv2.imread(train_set[_i]['img_path'], cv2.IMREAD_COLOR)
            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)  # 串连img
        imgs = imgs / 255.0
        mean, std = [], []
        for _j in range(3):
            mean.append(imgs[:, :, _j, :].mean())
            std.append(imgs[:, :, _j, :].std())
        return mean, std

    def __getitem__(self, idx):  # 返回单个样本的数据和标签，供DataLoader调用
        img_path, is_face, landmark = self.data_set[idx].values()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.transform is not None:  # 若有transform操作则进行transform
            img = self.transform(img)
        return img, label

    def __len__(self):  # 返回数据集大小
        return len(self.data_set)


class DataSet(data.Dataset):
    def __init__(self, transform=None):
        generate_data = GenerateImageData()
        self.data_save_path = generate_data.data_save_path  # 数据位置
        self.transform = transform  # 输入数据转换
        train_set, test_set = self.read_txt()  # 调用读取txt函数，得到数据字典
        print(len(train_set))
        print(len(test_set))
        if args.re_cal_mean_std:
            self.data_mean, self.data_std = self.cal_mean_std(train_set)
            print(f'data_mean: {self.data_mean}  data_std: {self.data_std}')
        else:
            self.data_mean, self.data_std = [0.382, 0.424, 0.525], [0.256, 0.260, 0.284]  # use cv2 BGR
        self.data_set = train_set if args.work in ['train', 'finetune'] else test_set  # 选定数据集

    def read_txt(self):  # 读取2个label.txt文件中的数据，返回以字典形式的存储
        train_test = []
        for sub_path in self.data_save_path:
            sub_data = []
            label_data = open(sub_path + 'label.txt')
            label_lines = label_data.readlines()
            label_data.close()
            for single_line in label_lines:
                single_line_split = single_line.split()
                img_path = single_line_split[0]
                is_face = single_line_split[1]
                landmarks = np.array([float(_i) for _i in single_line_split[1:]]) if is_face == 1 else np.array([])
                sub_data.append({'img_path': img_path, 'is_face': is_face, 'landmarks': landmarks})
            train_test.append(sub_data)
        return train_test

    @staticmethod
    def cal_mean_std(train_set):
        img = cv2.imread(train_set[0]['img_path'], cv2.IMREAD_COLOR)
        imgs = img[:, :, :, np.newaxis]
        for _i in range(1, len(train_set)):
            if _i % 500 == 0:
                print(_i)
            img = cv2.imread(train_set[_i]['img_path'], cv2.IMREAD_COLOR)
            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)  # 串连img
        imgs = imgs / 255.0
        mean, std = [], []
        for _j in range(3):
            mean.append(imgs[:, :, _j, :].mean())
            std.append(imgs[:, :, _j, :].std())
        return mean, std

    def __getitem__(self, idx):  # 返回单个样本的数据和标签，供DataLoader调用
        img_path, is_face, landmark = self.data_set[idx].values()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.transform is not None:  # 若有transform操作则进行transform
            img = self.transform(img)
        return img, label

    def __len__(self):  # 返回数据集大小
        return len(self.data_set)


class NetStage3(nn.Module):  # 模型定义
    def __init__(self):
        super(NetStage3, self).__init__()

        block1 = nn.Sequential()
        block1.add_module('conv1_1', nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=0))  # 54 * 54
        block1.add_module('prelu_conv1_1', nn.PReLU())
        block1.add_module('pool1', nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))  # 27 * 27
        self.block1 = block1

        block2 = nn.Sequential()
        block2.add_module('conv2_1', nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0))  # 25 * 25
        block2.add_module('prelu_conv2_1', nn.PReLU())
        block2.add_module('conv2_2', nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0))  # 23 * 23
        block2.add_module('prelu_conv2_2', nn.PReLU())
        block2.add_module('pool2', nn.AvgPool2d(kernel_size=2, stride=2, padding=1, ceil_mode=True))  # 12 * 12
        self.block2 = block2

        block3 = nn.Sequential()
        block3.add_module('conv3_1', nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=0))  # 10 * 10
        block3.add_module('prelu_conv3_1', nn.PReLU())
        block3.add_module('conv3_2', nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=0))  # 8 * 8
        block3.add_module('prelu_conv3_2', nn.PReLU())
        block3.add_module('pool3', nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))  # 4 * 4
        self.block3 = block3

        block4_base = nn.Sequential()
        block4_base.add_module('conv4_1', nn.Conv2d(24, 40, kernel_size=3, stride=1, padding=1))  # 4 * 4
        block4_base.add_module('prelu_conv4_1', nn.PReLU())
        self.block4_base = block4_base

        block4_class = nn.Sequential()  # 分类分支
        block4_class.add_module('conv4_2_cls', nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1))  # 4 * 4
        block4_class.add_module('prelu_conv4_2_cls', nn.PReLU())
        self.block4_class = block4_class

        block4_ldmk = nn.Sequential()  # 关键点检测分支
        block4_ldmk.add_module('conv4_2', nn.Conv2d(40, 80, kernel_size=3, stride=1, padding=1))  # 4 * 4
        block4_ldmk.add_module('prelu_conv4_2', nn.PReLU())
        self.block4_ldmk = block4_ldmk

        block5_class = nn.Sequential()  # 分类分支全连接层
        block5_class.add_module('ip1_cls', nn.Linear(640, 128))
        block5_class.add_module('prelu_ip1_cls', nn.PReLU())
        block5_class.add_module('ip2_cls', nn.Linear(128, 128))
        block5_class.add_module('prelu_ip2_cls', nn.PReLU())
        block5_class.add_module('ip3_cls', nn.Linear(128, 2))
        block5_class.add_module('face_score', nn.Softmax(dim=1))
        self.block5_class = block5_class

        block5_ldmk = nn.Sequential()  # 关键点检测全连接层
        block5_ldmk.add_module('ip1', nn.Linear(1280, 128))
        block5_ldmk.add_module('prelu_ip1', nn.PReLU())
        block5_ldmk.add_module('ip2', nn.Linear(128, 128))
        block5_ldmk.add_module('prelu_ip2', nn.PReLU())
        block5_ldmk.add_module('landmarks', nn.Linear(128, 42))
        self.block5_ldmk = block5_ldmk

    def forward(self, img):  # 前向传播
        # 共用
        result_block1 = self.block1(img)
        result_block2 = self.block2(result_block1)
        result_block3 = self.block3(result_block2)
        result_block4_base = self.block4_base(result_block3)
        # 分类
        result_block4_class = self.block4_class(result_block4_base)  # 分类分支
        fc_input_class = result_block4_class.view(result_block4_class.size(0), -1)  # 进行压平操作
        result_block5_class = self.block5_class(fc_input_class)
        # 关键点预测
        result_block4_ldmk = self.block4_ldmk(result_block4_base)  # 关键点检测分支
        fc_input_ldmk = result_block4_ldmk.view(result_block4_ldmk.size(0), -1)  # 进行压平操作
        result_block5_ldmk = self.block5_ldmk(fc_input_ldmk)
        return result_block5_class, result_block5_ldmk


def load_pretrained_netstage1(pretrained_model_dict, model):  # 关键点检测分支加载预训练模型参数
    mapping = [['block4.conv4_1.weight', 'block4_base.conv4_1.weight'],  # 预训练模型和当前模型模块名称映射表
               ['block4.conv4_1.bias', 'block4_base.conv4_1.bias'],
               ['block4.prelu_conv4_1.weight', 'block4_base.prelu_conv4_1.weight'],
               ['block4.conv4_2.weight', 'block4_ldmk.conv4_2.weight'],
               ['block4.conv4_2.bias', 'block4_ldmk.conv4_2.bias'],
               ['block4.prelu_conv4_2.weight', 'block4_ldmk.prelu_conv4_2.weight'],
               ['block5.ip1.weight', 'block5_ldmk.ip1.weight'],
               ['block5.ip1.bias', 'block5_ldmk.ip1.bias'],
               ['block5.prelu_ip1.weight', 'block5_ldmk.prelu_ip1.weight'],
               ['block5.ip2.weight', 'block5_ldmk.ip2.weight'],
               ['block5.ip2.bias', 'block5_ldmk.ip2.bias'],
               ['block5.prelu_ip2.weight', 'block5_ldmk.prelu_ip2.weight'],
               ['block5.landmarks.weight', 'block5_ldmk.landmarks.weight'],
               ['block5.landmarks.bias', 'block5_ldmk.landmarks.bias']]
    new_model_static_dict = model.state_dict()
    for k, v in pretrained_model_dict.items():  # 查找相同键名或存在于mapping列表中键名的模块，更换参数
        if k in new_model_static_dict.keys():
            new_model_static_dict[k] = v
        else:
            for i in mapping:
                if k == i[0]:
                    new_model_static_dict[i[1]] = v
    model.load_state_dict(new_model_static_dict)
    return model


class MultiWorks:  # 多任务
    def __init__(self, load_model_path=None):
        self.start_time = time.time()  # 开始时间，用于输出时长
        self.load_model_path = load_model_path  # 微调、测试和预测时提供模型加载路径
        # 数据集
        self.data_mean, self.data_std = DataSet().data_mean, DataSet().data_std
        self.data_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(tuple(self.data_mean), tuple(self.data_std))])
        self.data_set = DataSet(transform=self.data_transform)

        # 执行选择任务，train; test; finetune; predict;
        if args.work not in ['train', 'test', 'finetune', 'predict']:
            print("The args.work should be one of ['train', 'test', 'finetune', 'predict']")
        elif args.work == "train":  # 训练
            self.train()
        elif self.load_model_path is None:
            print("Please input 'load_model_path'")
        elif args.work == "test":  # 测试
            self.test()
        elif args.work == "finetune":  # 调模型
            self.finetune()
        elif args.work == "predict":  # 预测
            self.predict()

    def train(self):
        data_loader = torch.utils.data.DataLoader(self.data_set, batch_size=args.batch_size, shuffle=True)
        # 输出数据集大小
        print(f"Start Train!    len_data_set: {self.data_set.__len__()}")

        model = NetStage3()
        for m in model.modules():  # 卷积层参数初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)  # kaiming初始化
                m.bias.data.fill_(0)  # 偏差初始为零
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_()  # 全连接层参数初始化

        if args.load_pretrained_model:  # 加载预训练的stage1网络参数
            pretrained_model = torch.load(args.model_stage1_path)
            model = load_pretrained_netstage1(pretrained_model, model).to(device)

        # active_weight_list = ['block4_class.conv4_2_cls.weight', 'block4_class.conv4_2_cls.bias',
        #                       'block4_class.prelu_conv4_2_cls.weight', 'block5_class.ip1_cls.weight',
        #                       'block5_class.ip1_cls.bias', 'block5_class.prelu_ip1_cls.weight',
        #                       'block5_class.ip2_cls.weight', 'block5_class.ip2_cls.bias',
        #                       'block5_class.prelu_ip2_cls.weight', 'block5_class.ip3_cls.weight',
        #                       'block5_class.ip3_cls.bias']
        # for name, param in model.named_parameters():  # 仅保留分类分支参数梯度更新
        #     if name not in active_weight_list:
        #         param.requires_grad = False

        criterion_class = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(
            [args.negative_weight, args.positive_weight], dtype='float32')).to(device))  # 交叉熵引入权重
        criterion_ldmk = nn.MSELoss()
        current_lr = args.lr * 1000  # 步长
        optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum)  # 优化器

        # 采集loss存储为.csv
        collect_loss = [['epoch', 'lr', 'mean_loss_class', 'max_loss_class', 'min_loss_class',
                         'tp', 'tn', 'fp', 'fn', 'mean_precision', 'mean_recall',
                         'mean_loss_ldmk', 'max_loss_ldmk', 'min_loss_ldmk',
                         'mean_total_loss', 'max_total_loss', 'min_total_loss']]
        epoch_count = []
        loss_record = []
        precision_recall = []
        cost_time_record = []
        for i in range(args.epochs):  # 开始训练
            epoch_class_loss, tp, tn, fp, fn, epoch_ldmk_loss, epoch_total_loss = [], [], [], [], [], [], []  # loss
            for index, (img, is_face, label) in enumerate(data_loader):
                img = img.to(device)  # 图像输入设备
                label = torch.from_numpy(label.numpy().astype('float32')).to(device)  # 标签输入设备
                is_face = is_face.to(device)  # 数据输入设备
                optimizer.zero_grad()  # 优化器梯度清零
                output_class, output_ldmk = model(img)  # 计算模型输出
                loss_class = criterion_class(output_class, is_face)  # 分类loss
                if output_ldmk[is_face == 1].numel():  # 检测非空
                    loss_ldmk = criterion_ldmk(output_ldmk[is_face == 1], label[is_face == 1])  # 回归loss
                    epoch_ldmk_loss.append(loss_ldmk.item())
                    loss = args.lamda_class_weight * loss_class + loss_ldmk  # 整体loss
                else:
                    loss = args.lamda_class_weight * loss_class  # 整体loss

                # 采集epoch内的loss
                epoch_class_loss.append(loss_class.item())
                epoch_total_loss.append(loss.item())
                class_result = torch.argmax(output_class, dim=1)  # 返回分类结果
                tp.append(torch.sum(class_result[is_face == 1] == 1).item())  # 记录分类情况
                tn.append(torch.sum(class_result[is_face == 0] == 0).item())
                fp.append(torch.sum(class_result[is_face == 0] == 1).item())
                fn.append(torch.sum(class_result[is_face == 1] == 0).item())

                loss.backward()  # loss值对模型内参数进行反向传播
                optimizer.step()  # 优化器更新模型参数
            total_loss = sum(epoch_total_loss) / (len(epoch_total_loss))
            class_loss = sum(loss_class) / (len(loss_class))
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            # 供visdom显示
            epoch_count.append(i + 1)
            loss_record.append([total_loss, class_loss])
            precision_recall.append([precision, recall])
            cost_time_record.append(time.time() - self.start_time)
            vis.line(X=epoch_count, Y=loss_record, win='chart1', opts=opts1)
            vis.line(X=epoch_count, Y=precision_recall, win='chart2', opts=opts2)
            vis.line(X=epoch_count, Y=cost_time_record, win='chart3', opts=opts3)
            collect_loss = self.collect_print(self, collect_loss, epoch_class_loss, tp, fp, fn, epoch_ldmk_loss,
                                              epoch_total_loss, epoch=i, lr=current_lr)  # 打印结果
            # 保存模型和loss
        if args.save_model:  # 是否保存模型
            if not os.path.exists(args.save_directory):  # 新建保存文件夹
                os.makedirs(args.save_directory)
            save_model_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M') + '_train_epoch_' + str(i)
                                           + ".pt")  # 保存路径
            save_loss_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M') + '_train_loss.csv')
            torch.save(model.state_dict(), save_model_path)
            self.writelist2csv(collect_loss, save_loss_path)
            print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Train complete!')

    def test(self):
        do_data_transform = transforms.Compose([transforms.ToTensor()])
        data_set = DataSet(args.train_set_ratio, transform=do_data_transform)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.test_batch_size, shuffle=False)
        # 输出数据集大小
        print(f"Start Test!      len_data_set: {len(data_set.data_set)}")
        model = NetStage3().to(device)
        model.load_state_dict(torch.load(self.load_model_path))
        model.eval()

        criterion_class = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(
            [args.negative_weight, args.positive_weight], dtype='float32')).to(device))  # 交叉熵引入权重
        criterion_ldmk = nn.MSELoss()

        # 采集loss存储为.csv
        collect_loss = [['epoch', 'lr', 'mean_loss_class', 'max_loss_class', 'min_loss_class',
                         'tp', 'tn', 'fp', 'fn', 'mean_precision', 'mean_recall',
                         'mean_loss_ldmk', 'max_loss_ldmk', 'min_loss_ldmk',
                         'mean_total_loss', 'max_total_loss', 'min_total_loss']]

        epoch_class_loss, tp, tn, fp, fn, epoch_ldmk_loss, epoch_total_loss = [], [], [], [], [], [], []
        for index, (img, is_face, label) in enumerate(data_loader):
            img_plot, label_plt = img.numpy(), label.numpy()
            img = img.to(device)  # 图像输入设备
            label = torch.from_numpy(label.numpy().astype('float32')).to(device)  # 标签输入设备
            is_face = is_face.to(device)  # 数据输入设备
            output_class, output_ldmk = model(img)  # 计算模型输出
            loss_class = criterion_class(output_class, is_face)  # 计算loss值

            if output_ldmk[is_face == 1].numel():
                loss_ldmk = criterion_ldmk(output_ldmk[is_face == 1], label[is_face == 1])
                epoch_ldmk_loss.append(loss_ldmk.item())  # 采集epoch内的loss
                loss = args.lamda_class_weight * loss_class + loss_ldmk
            else:
                loss = args.lamda_class_weight * loss_class
            epoch_class_loss.append(loss_class.item())  # 采集epoch内的loss
            epoch_total_loss.append(loss.item())
            class_result = torch.argmax(output_class, dim=1)  # 返回分类结果
            tp.append(torch.sum(class_result[is_face == 1] == 1).item())  # 记录分类情况
            tn.append(torch.sum(class_result[is_face == 0] == 0).item())
            fp.append(torch.sum(class_result[is_face == 0] == 1).item())
            fn.append(torch.sum(class_result[is_face == 1] == 0).item())

            # self.show_face_landmarks(img_plot, output_class.cpu(), output_ldmk.cpu(),
            #                          is_face=is_face.cpu(), labels=label_plt, plots=1)     # 绘制真实值与预测结果
        collect_loss = self.collect_print(self, collect_loss, epoch_class_loss, tp, fp, fn, epoch_ldmk_loss,
                                          epoch_total_loss)  # 打印结果
        # 保存loss
        save_loss_path = self.load_model_path[:-3] + '_test_result.csv'
        self.writelist2csv(collect_loss, save_loss_path)
        print(f'--Save complete!\n--save_loss_path: {save_loss_path}')
        print('Test complete!')

    def finetune(self):
        do_data_transform = transforms.Compose([transforms.ToTensor()])
        data_set = DataSet(args.train_set_ratio, transform=do_data_transform)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.test_batch_size, shuffle=False)
        # 输出数据集大小
        print(f"Start Finetune!  len_data_set: {len(data_set.data_set)}")
        model = NetStage3().to(device)
        model.load_state_dict(torch.load(self.load_model_path))

        criterion_class = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(
            [args.negative_weight, args.positive_weight], dtype='float32')).to(device))  # 交叉熵引入权重
        criterion_ldmk = nn.MSELoss()
        current_lr = args.lr * 100  # 步长
        optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum * 0)  # 优化器

        # 采集loss存储为.csv
        collect_loss = [['epoch', 'lr', 'mean_loss_class', 'max_loss_class', 'min_loss_class',
                         'tp', 'tn', 'fp', 'fn', 'mean_precision', 'mean_recall',
                         'mean_loss_ldmk', 'max_loss_ldmk', 'min_loss_ldmk',
                         'mean_total_loss', 'max_total_loss', 'min_total_loss']]
        for i in range(args.epochs):  # 开始训练
            epoch_class_loss, tp, tn, fp, fn, epoch_ldmk_loss, epoch_total_loss = [], [], [], [], [], [], []
            for index, (img, is_face, label) in enumerate(data_loader):
                img = img.to(device)  # 图像输入设备
                label = torch.from_numpy(label.numpy().astype('float32')).to(device)  # 标签输入设备
                is_face = is_face.to(device)  # 数据输入设备
                optimizer.zero_grad()  # 优化器梯度清零
                output_class, output_ldmk = model(img)  # 计算模型输出
                loss_class = criterion_class(output_class, is_face)  # 分类loss
                if output_ldmk[is_face == 1].numel():  # 检测非空
                    loss_ldmk = criterion_ldmk(output_ldmk[is_face == 1], label[is_face == 1])  # 回归loss
                    epoch_ldmk_loss.append(loss_ldmk.item())
                    loss = args.lamda_class_weight * loss_class + loss_ldmk  # 整体loss
                else:
                    loss = args.lamda_class_weight * loss_class  # 整体loss

                epoch_class_loss.append(loss_class.item())  # 采集epoch内的loss
                epoch_total_loss.append(loss.item())  # 采集epoch内的loss
                class_result = torch.argmax(output_class, dim=1)  # 返回分类结果
                tp.append(torch.sum(class_result[is_face == 1] == 1).item())  # 记录分类情况
                tn.append(torch.sum(class_result[is_face == 0] == 0).item())
                fp.append(torch.sum(class_result[is_face == 0] == 1).item())
                fn.append(torch.sum(class_result[is_face == 1] == 0).item())
                loss.backward()  # loss值对模型内参数进行反向传播
                optimizer.step()  # 优化器更新模型参数

            collect_loss = self.collect_print(self, collect_loss, epoch_class_loss, tp, fp, fn, epoch_ldmk_loss,
                                              epoch_total_loss, i=i, current_lr=current_lr)  # 打印结果
            # 保存模型和loss
        if args.save_model:  # 是否保存模型
            if not os.path.exists(args.save_directory):  # 新建保存文件夹
                os.makedirs(args.save_directory)
            save_finetune_model_path = self.load_model_path[:-3] + '_finetune_' + str(i) + ".pt"
            save_finetune_loss_path = self.load_model_path[:-3] + '_finetune_' + str(i) + "_loss.csv"
            torch.save(model.state_dict(), save_finetune_model_path)
            self.writelist2csv(collect_loss, save_finetune_loss_path)
            print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Finetune complete!')

    def predict(self):
        # 输出数据集大小
        print(f"Start Predict!")
        model = NetStage3().to(device)
        model.load_state_dict(torch.load(self.load_model_path))
        model.eval()

        predict_result = [['output_class', 'class_result', 'output_ldmk']]
        train_set_mean, train_set_std = 114.63897582368799, 66.18579280506815
        while True:
            img_ori, k = self.capture_predict_face()
            if k == 27:
                print('Quit Predict!')
                break
            img_norm = ((img_ori - train_set_mean) / (train_set_std + 0.0000001)).astype('float32')
            img_tensor = transforms.ToTensor()(img_norm)
            img_s = np.expand_dims(img_tensor, axis=0)
            img = torch.from_numpy(img_s).to(device)  # 数据输入设备
            output_class, output_ldmk = model(img)  # 前向传播计算模型输出
            class_result = torch.argmax(output_class, dim=1)  # 返回分类结果
            predict_result.append([output_class.data, class_result.data, output_ldmk.data])
            if class_result == 1:
                print('Detect face!')
                self.show_face_landmarks(img.cpu(), output_class.cpu(), output_ldmk.cpu())  # 查看预测结果
        save_eval_loss = self.load_model_path[:-3] + '_predict_result.csv'
        self.writelist2csv(predict_result, save_eval_loss)
        print('Predict complete!')

    def collect_print(self, collect_loss, epoch_class_loss, tp, fp, fn, epoch_ldmk_loss, epoch_total_loss,
                      epoch=0, lr=0):
        mean_loss_class = sum(epoch_class_loss) / len(epoch_class_loss)
        max_loss_class = max(epoch_class_loss)
        min_loss_class = min(epoch_class_loss)
        mean_precision = sum(tp) * 1.0 / ((sum(tp) + sum(fp)) * 1.0)
        mean_recall = sum(tp) * 1.0 / ((sum(tp) + sum(fn)) * 1.0)
        mean_loss_ldmk = sum(epoch_ldmk_loss) / len(epoch_ldmk_loss)
        max_loss_ldmk = max(epoch_ldmk_loss)
        min_loss_ldmk = min(epoch_ldmk_loss)
        mean_total_loss = sum(epoch_total_loss) / len(epoch_total_loss)
        max_total_loss = max(epoch_total_loss)
        min_total_loss = min(epoch_total_loss)
        collect_loss.append([epoch, lr, mean_loss_class, max_loss_class, min_loss_class, sum(tp), sum(tn),
                             sum(fp), sum(fn), mean_precision, mean_recall, mean_loss_ldmk, max_loss_ldmk,
                             min_loss_ldmk, mean_total_loss, max_total_loss, min_total_loss])  # 采集loss
        # 打印结果
        print(f'epoch: {i}  cost_time: {time.time() - self.start_time}\n--mean_loss_class: {mean_loss_class}  '
              f'max_loss_class: {max_loss_class}  min_loss_class: {min_loss_class}\n--tp: {sum(tp)}  fp: {sum(fp)}'
              f'  tn: {sum(tn)}  fn: {sum(fn)}\n--mean_precision: {mean_precision}  mean_recall: {mean_recall}')
        return collect_loss

    @staticmethod
    def writelist2csv(list_data, csv_name):  # 列表写入.csv
        with open(csv_name, "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for one_slice in list_data:
                csv_writer.writerow(one_slice)

    @staticmethod
    def show_face_landmarks(imgs, output_class, out_ldmk, is_face=None, labels=None, plots=1):  # 绘制处理后的图片
        random_plt = random.sample(range(len(imgs)), plots)  # plots每个patch选择绘制的图片数
        for i in random_plt:
            img = imgs[i][0]
            plt.imshow(img, cmap='gray')
            class_result = output_class[i]
            title_str = ''
            landmarks_o = out_ldmk[i].detach().numpy().reshape(-1, 2).T
            if is_face is not None:
                if is_face[i]:
                    title_str = title_str + 'Is face. blue-label. '
                    landmarks_l = labels[i].reshape(-1, 2).T
                    plt.scatter(landmarks_l[0], landmarks_l[1], s=3, c='b')
                else:
                    title_str = title_str + 'Is not face. '
                if torch.argmax(class_result):
                    plt.scatter(landmarks_o[0], landmarks_o[1], s=3, c='r')
                    title_str = title_str + 'Predict is 1. red-predict'
                else:
                    title_str = title_str + 'Predict is 0'
            else:
                if torch.argmax(class_result):
                    plt.scatter(landmarks_o[0], landmarks_o[1], s=3, c='r')
                    title_str = title_str + 'Unknow. Predict is 1. red-predict'
                else:
                    title_str = title_str + 'Unknow. Predict is 0'
            plt.title(title_str)
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--train-set-ratio', type=float, default=0.8, metavar='N',
                        help='train percentage of all data (default: 0.8)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.00000001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--load-pretrained-model', type=bool, default=True,
                        help='load pretrained stage1 model')
    parser.add_argument('--model-stage1-path',
                        type=str, default='save_stage1_model/202004011659_train_epoch_199_finetune_299.pt',
                        help='model_stage_1 path, for load')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='save_stage3_model',
                        help='learnt models are saving here')
    parser.add_argument('--positive-weight', type=float, default=2.0,
                        help='face positive class weight')
    parser.add_argument('--negative-weight', type=float, default=1.0,
                        help='face negative class weight')
    parser.add_argument('--lamda-class-weight', type=float, default=5.0,
                        help='class weight')
    parser.add_argument('--re-generate-data', type=bool, default=True,
                        help='if need to re generate data')
    parser.add_argument('--re-cal-mean-std', type=bool, default=False,
                        help='if need to re calculate dateset mean and std')
    parser.add_argument('--work', type=str, default='train',  # train, test, finetune, predict
                        help='training, test, predicting or finetuning')
    args = parser.parse_args()

    # visdom可视化设置
    vis = Visdom(env="face key point detection stage3")
    assert vis.check_connection()
    opts1 = {
        "title": 'total_loss of mean/max/min in epoch',
        "xlabel": 'epoch',
        "ylabel": 'loss',
        "width": 600,
        "height": 400,
        "legend": ['total_loss', 'class_loss']
    }
    opts2 = {
        "title": 'precision and recall with epoch',
        "xlabel": 'epoch',
        "ylabel": 'percentage',
        "width": 400,
        "height": 300,
        "legend": ['precision', 'recall']
    }
    opts3 = {
        "title": 'cost time with epoch',
        "xlabel": 'epoch',
        "ylabel": 'time in second',
        "width": 400,
        "height": 300,
        "legend": ['cost time']
    }

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # assignment = MultiWorks(load_model_path='save_stage3_model/stage3.pt')
    # torch.cuda.empty_cache()
    GenerateImageData()
