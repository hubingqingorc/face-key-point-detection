# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
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
import collections
import shutil
from visdom import Visdom
import multiprocessing
from datetime import datetime


class GenerateImageData:
    def __init__(self, generate=True):
        self.data_path = ['data/I/', 'data/II/']  # 数据位置
        self.expand_ratio = 0.25  # 人脸框扩张率
        self.train_set_ratio = 0.8  # 训练集占比
        self.boundary_size = 112  # 裁剪图片尺寸
        self.shuffle_seed = 42  # 数据集洗牌随机种子
        self.flip_p = 1  # 对图片进行水平翻转的概率
        self.rotation_number = 4  # 单张图片旋转次数
        self.angle_interval = 5  # 每次旋转角度差
        self.data_save_path = ['stage2_train_data/', 'stage2_test_data/']  # 数据位置
        if generate:  # 生成图片集
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

    def split_train_test(self):  # 分割训练集和测试集
        np.random.seed(self.shuffle_seed)  # 输入随机种子
        shuffled_data = np.random.permutation(self.data_dic)
        train_size = int(len(self.data_dic) * self.train_set_ratio)  # 根据训练集占比得到训练集大小
        train_set, test_set = shuffled_data[:train_size], shuffled_data[train_size:]
        train_set = self.augment(train_set)
        test_set = [{'img_path': orig['img_path'], 'frame_corner': orig['frame_corner'],
                     'landmarks': orig['landmarks'], 'flip': 0, 'rotation': 0} for orig in test_set]
        return train_set, test_set

    def augment(self, train_set):
        augment_set = []
        even_rotation_number = self.rotation_number if self.rotation_number % 2 == 0 else self.rotation_number + 1
        rotation_degree_list = list(range(int(-even_rotation_number / 2), int(even_rotation_number / 2) + 1))
        # 增加旋转样本、水平翻转样本
        for orig in train_set:
            for i in rotation_degree_list:
                augment_set.append({'img_path': orig['img_path'], 'frame_corner': orig['frame_corner'],
                                    'landmarks': orig['landmarks'], 'flip': 0, 'rotation': i * self.angle_interval})
            # 随机水平翻转
            if random.random() <= self.flip_p:
                augment_set.append({'img_path': orig['img_path'], 'frame_corner': orig['frame_corner'],
                                    'landmarks': orig['landmarks'], 'flip': 1, 'rotation': 0})
        return augment_set

    def generate_data(self):  # 生成图片、标签数据并存入硬盘，供后续调用
        for _f in self.data_save_path:  # make directory
            if os.path.exists(_f):  # 若已存在，先删除文件夹
                shutil.rmtree(_f)
            time.sleep(3)  # 为完全删除文件夹留3s缓冲时间
            os.makedirs(_f)

        # 生成训练集、测试集数据
        data_set = [self.train_set, self.test_set]
        for _s in range(2):
            file = open(self.data_save_path[_s] + 'label.txt', 'w')  # 生成训练集
            name_list = []
            for _i in data_set[_s]:
                # 生成存储时文件名
                name = _i['img_path'].split("/", 2)[-1]
                name_list.append(name)
                name_count = collections.Counter(name_list)
                name_times = name_count[name]
                name_split = name.split('.')
                img_save_path = self.data_save_path[_s] + name_split[0] + '_' + str(name_times) + '.' + name_split[1]
                # 图片处理：旋转 + 剪裁 + 水平翻转 + 等比例放缩
                if _i['rotation'] == 0:
                    img = cv2.imread(_i['img_path'], cv2.IMREAD_COLOR)
                    label = _i['landmarks'].reshape(-1, 2)
                else:
                    img, label = self.img_rotate(_i)
                img, label = self.expand_crop(img, label, _i['frame_corner'])  # 返回扩大人脸框并剪裁的数据
                if _i['flip'] == 1:
                    img, label = self.img_flip(img, label)
                img, label = self.proportion_resize(img, label)  # 保持x y轴比例进行resize
                # 存储图片及写入label.txt
                str_path_label = img_save_path
                for _j in label:
                    str_path_label = str_path_label + ' ' + str(_j)
                file.write(str_path_label + chr(10))
                cv2.imwrite(img_save_path, img)
            file.close()

    @staticmethod
    def img_rotate(single_face_data):
        img = cv2.imread(single_face_data['img_path'], cv2.IMREAD_COLOR)
        frame_corner = single_face_data['frame_corner']
        alpha = single_face_data['rotation']
        frame_center = ((frame_corner[0] + frame_corner[2]) / 2.0, (frame_corner[1] + frame_corner[3]) / 2.0)
        m = cv2.getRotationMatrix2D(frame_center, alpha, 1)
        img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))
        rotation_matrix = np.array([[math.cos(alpha / 180 * math.pi), -math.sin(alpha / 180 * math.pi)],
                                    [math.sin(alpha / 180 * math.pi), math.cos(alpha / 180 * math.pi)]])
        label = (single_face_data['landmarks'].reshape(-1, 2) - np.array(frame_center)).dot(rotation_matrix) \
                + np.array(frame_center)
        return img, label

    def expand_crop(self, img, label, frame_corner):  # 对单张图片扩大人脸框，返回剪裁后结果及相对坐标
        height, width, _ = img.shape
        roi_x1, roi_y1, roi_x2, roi_y2 = frame_corner
        # 计算水平方向和垂直方向扩张尺寸
        width_expand, height_expand = (roi_x2 - roi_x1) * self.expand_ratio, (roi_y2 - roi_y1) * self.expand_ratio
        e_roi_x1 = int(max(0, roi_x1 - width_expand))
        e_roi_y1 = int(max(0, roi_y1 - height_expand))
        e_roi_x2 = int(min(width - 1, roi_x2 + width_expand))
        e_roi_y2 = int(min(height - 1, roi_y2 + height_expand))
        e_roi_x1_y1 = np.array([e_roi_x1, e_roi_y1])
        relative_landmarks = label - e_roi_x1_y1  # 计算相对于左上角的坐标
        img_crop = img[e_roi_y1:e_roi_y2, e_roi_x1:e_roi_x2]
        return img_crop, relative_landmarks

    @staticmethod
    def img_flip(img, label):
        img_flip = cv2.flip(img, 1)
        label = label * np.array([-1, 1]) + np.array([img_flip.shape[1], 0])
        bef_flip, aft_flip = [1, 2, 3, 8, 17, 10, 13, 21], [6, 5, 4, 9, 18, 7, 11, 20]  # 图片水平翻转后关键点交换
        for i in range(len(bef_flip)):
            temp = label[bef_flip[i] - 1].copy()
            label[bef_flip[i] - 1] = label[aft_flip[i] - 1]
            label[aft_flip[i] - 1] = temp
        return img_flip, label

    def proportion_resize(self, img, label):  # 保持x y轴比例进行resize
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
        resize_label = label * ratio + np.array([left, top])  # 计算填充后关键点位置
        label = resize_label.flatten()  # 对关键点进行压平操作
        return cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_REPLICATE), label


class DataSet(data.Dataset):
    def __init__(self, is_train=True, re_generate_data=False, transform=None):
        generate_data = GenerateImageData(generate=re_generate_data)
        self.data_save_path = generate_data.data_save_path  # 数据位置
        self.transform = transform  # 输入数据转换
        train_set, test_set = self.read_txt()  # 调用读取txt函数，得到数据字典
        if args.re_cal_mean_std :
            self.data_mean, self.data_std = self.cal_mean_std(train_set)
            print(f'data_mean: {self.data_mean}  data_std: {self.data_std}')
        else:
            self.data_mean, self.data_std = [0.380, 0.422, 0.523], [0.256, 0.260, 0.285]  # use cv2 BGR
        self.data_set = train_set if is_train else test_set  # 选定数据集

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
                landmarks = np.array([float(_i) for _i in single_line_split[1:]])
                sub_data.append({'img_path': img_path, 'landmarks': landmarks})
            train_test.append(sub_data)
        return train_test

    @staticmethod
    def cal_mean_std(train_set):
        img = cv2.imread(train_set[0]['img_path'], cv2.IMREAD_COLOR)
        imgs = img[:, :, :, np.newaxis]
        for _i in range(1, len(train_set)):
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
        img_path, label = self.data_set[idx].values()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.transform is not None:  # 若有transform操作则进行transform
            img = self.transform(img)
        return img, label

    def __len__(self):  # 返回数据集大小
        return len(self.data_set)


def conv3x3(in_chnls, out_chnls, stride=1):
    return nn.Conv2d(in_chnls, out_chnls, kernel_size=3, stride=stride, padding=1)


class ResidualBlock(nn.Module):    # ref 《Training and investigating Residual Nets》
    def __init__(self, in_chnls, out_chnls, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv_1 = conv3x3(in_chnls, out_chnls, stride=stride)
        self.bn_1 = nn.BatchNorm2d(out_chnls)
        self.prelu_1 = nn.PReLU()
        self.conv_2 = conv3x3(out_chnls, out_chnls)        
        self.downsample = downsample
        self.bn_2 = nn.BatchNorm2d(out_chnls)
        self.prelu_2 = nn.PReLU()

    def forward(self, x):
        residual = x
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.prelu_1(x)
        x = self.conv_2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.bn_2(x)
        return self.prelu_2(x)


class NetStage2(nn.Module):  # 模型定义
    def __init__(self):
        super(NetStage2, self).__init__()

        block1 = nn.Sequential()
        block1.add_module('conv1_1', nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2))  # single_img: 56 * 56
        block1.add_module('prelu_conv1_1', nn.PReLU())
        self.block1 = block1

        block2 = nn.Sequential()
        block2.add_module('conv2_1', ResidualBlock(8, 8, stride=1))  # single_img: 56 * 56
        block2.add_module('conv2_2', ResidualBlock(8, 8, stride=1))  # single_img: 56 * 56
        self.block2 = block2

        block3 = nn.Sequential()
        downsample_3 = nn.Sequential(nn.Conv2d(8, 32, kernel_size=1, stride=2))
        block3.add_module('conv3_1', ResidualBlock(8, 32, stride=2, downsample=downsample_3))  # single_img: 28 * 28
        block3.add_module('conv3_2', ResidualBlock(32, 32, stride=1))  # single_img: 28 * 28
        self.block3 = block3

        block4 = nn.Sequential()
        downsample_4 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, stride=2))
        block4.add_module('conv4_1', ResidualBlock(32, 64, stride=2, downsample=downsample_4))  # single_img: 14 * 14
        block4.add_module('conv4_2', ResidualBlock(64, 64, stride=1))  # single_img: 14 * 14
        self.block4 = block4

        block5 = nn.Sequential()
        downsample_5 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2))
        block5.add_module('conv5_1', ResidualBlock(64, 128, stride=2, downsample=downsample_5))  # single_img: 7 * 7
        block5.add_module('conv5_2', ResidualBlock(128, 128, stride=1))  # single_img: 7 * 7
        self.block5 = block5

        block_fc = nn.Sequential()
        block_fc.add_module('fc1', nn.Linear(128 * 7 * 7, 128))
        block_fc.add_module('fc_prelu_1', nn.PReLU())
        block_fc.add_module('fc2', nn.Linear(128, 128))
        block_fc.add_module('fc_prelu_2', nn.PReLU())
        block_fc.add_module('fc_dropout', nn.Dropout(0.7))
        block_fc.add_module('fc3', nn.Linear(128, 42))
        self.block_fc = block_fc

    def forward(self, x):  # 前向传播
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        fc_input = x.view(x.size(0), -1)  # 进行压平操作
        x = self.block_fc(fc_input)
        return x


class MultiWorks:  # 多任务
    def __init__(self, load_model_path=None):
        self.start_time = time.time()  # 开始时间，用于输出时长
        self.load_model_path = load_model_path  # 微调、测试和预测时提供模型加载路径        
        # 数据集
        self.data_mean, self.data_std = DataSet(re_generate_data=False).data_mean, DataSet().data_std
        self.data_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(tuple(self.data_mean), tuple(self.data_std))])

        if not os.path.exists(args.save_directory):  # 新建模型保存文件夹
            os.makedirs(args.save_directory)

        # 执行选择任务，train; test; finetune; predict;
        if args.work not in ['train', 'test', 'finetune', 'predict']:
            print("The args.work should be one of ['train', 'test', 'finetune', 'predict']")
        elif args.work == "train":  # 训练
            self.train()
        elif self.load_model_path is None:
            print("Please input 'load_model_path'")
        elif args.work == "finetune":  # 调模型
            self.finetune()
        elif args.work == "test":  # 测试
            mean_loss, max_loss, min_loss = self.test(self.load_model_path, is_path=True)
            print(f'mean_loss: {mean_loss}  max_loss: {max_loss}  min_loss: {min_loss}'
                  f'cost_time: {time.time() - self.start_time}')
            collect_loss = (['mean_loss', 'max_loss', 'min_loss'], [mean_loss, max_loss, min_loss])  # 采集loss
            save_loss_path = self.load_model_path[:-3] + '_test_result.csv'
            self.writelist2csv(collect_loss, save_loss_path)
            print(f'--Save complete!\n--save_loss_path: {save_loss_path}\n')
            print('Test complete!')
        elif args.work == "predict":  # 预测
            self.predict()

    def train(self):
        data_set = DataSet(re_generate_data=args.re_generate_data, transform=self.data_transform)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
        # 输出开始及数据集大小
        print(f"Start Train!  len_dataset: {data_set.__len__()}")
        model = NetStage2().to(device).train()
        for m in model.modules():  # 参数初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)  # 卷积层kaiming初始化
                m.bias.data.fill_(0)  # 偏差初始为零
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_()  # 全连接层随机初始化
        criterion = nn.MSELoss()
        current_lr = args.lr  # 步长
        optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, betas=(0.9, 0.999))
        # optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum, weight_decay=0.5)  # 优化器

        collect_loss = [['epoch', 'current_lr', 'epoch_mean_loss', 'epoch_max_loss', 'epoch_min_loss',
                         'test_mean_loss', 'test_max_loss', 'test_min_loss']]
        epoch_count = []
        loss_record = []
        cost_time_record = []
        # latest_test_loss = []
        # break_label = False
        for i in range(args.epochs):  # 开始训练
            epoch_loss = []  # 每个epoch的loss
            for index, (img, label) in enumerate(data_loader):
                img = img.to(device)  # 图片输入设备
                label = torch.from_numpy(label.numpy().astype('float32')).to(device)  # 标签输入设备
                optimizer.zero_grad()  # 优化器梯度清零
                output = model(img)  # 计算模型输出
                loss = criterion(output, label)  # 计算loss值
                epoch_loss.append(loss.item())  # 采集epoch内的loss
                loss.backward()  # loss值对模型内参数进行反向传播
                optimizer.step()  # 优化器更新模型参数
            if i % 100 == 0:
                self.show_face_landmarks(img, i, labels=label, outputs=output)
            epoch_mean_loss = sum(epoch_loss) / (len(epoch_loss))
            epoch_max_loss = max(epoch_loss)
            epoch_min_loss = min(epoch_loss)
            # Early Stop 每个epoch存储模型并检验
            mean_loss, max_loss, min_loss = self.test(model.state_dict())
            # latest_test_loss.append(mean_loss)
            # if len(latest_test_loss) > 50:
            #     if latest_test_loss[-1] > max(latest_test_loss[:-1]):
            #         break_label = True
            # 供visdom显示
            epoch_count.append(i + 1)
            loss_record.append([epoch_mean_loss, epoch_max_loss, epoch_min_loss, mean_loss, max_loss, min_loss])
            cost_time_record.append(time.time() - self.start_time)
            vis.line(X=epoch_count, Y=loss_record, win='chart1', opts=opts1)
            vis.line(X=epoch_count, Y=cost_time_record, win='chart2', opts=opts2)

            # if i == 5:
            #     current_lr = args.lr * 10
            #     optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum, weight_decay=0.5)  # 优化函数
            # if i == 15:
            #     current_lr = args.lr * 50
            #     optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum, weight_decay=0.5)  # 优化函数
            collect_loss.append([i, current_lr, epoch_mean_loss, epoch_max_loss, epoch_min_loss,
                                 mean_loss, max_loss, min_loss])  # 采集loss
            # if break_label:
            #     break
        # 保存模型和loss
        save_model_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M', time.localtime(self.start_time))
                                       + '_train_epoch_' + str(i) + ".pt")
        save_loss_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M', time.localtime(self.start_time))
                                      + '_train_loss.csv')
        torch.save(model.state_dict(), save_model_path)
        self.writelist2csv(collect_loss, save_loss_path)
        print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Train complete!')

    def test(self, input_model, is_path=False):
        data_set = DataSet(is_train=False, re_generate_data=args.re_generate_data, transform=self.data_transform)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.test_batch_size, shuffle=False)

        # 输出数据集大小
        if is_path:
            print(f"Start Test!  len_dataset: {data_set.__len__()}")
        model = NetStage2().to(device_1)
        if is_path:  # 模型参数加载
            model.load_state_dict(torch.load(input_model))
        else:
            model.load_state_dict(input_model)
        model.eval()  # 关闭参数梯度

        criterion = nn.MSELoss()  # SmoothL1Loss()
        epoch_loss = []  # 每个epoch的loss
        with torch.no_grad():
            for index, (img, label) in enumerate(data_loader):
                img = img.to(device_1)  # 图片输入设备
                label = torch.from_numpy(label.numpy().astype('float32')).to(device_1)  # 标签输入设备
                output = model(img)  # 前向传播计算模型输出
                loss = criterion(output, label)  # 计算loss值
                epoch_loss.append(loss.item())
                # 查看模型输出及标签
                # self.show_face_landmarks(img, 0, labels=label, outputs=output)
        epoch_mean_loss = sum(epoch_loss) / (len(epoch_loss))
        epoch_max_loss = max(epoch_loss)
        epoch_min_loss = min(epoch_loss)
        return epoch_mean_loss, epoch_max_loss, epoch_min_loss

    def finetune(self):
        data_set = DataSet(re_generate_data=False, transform=self.data_transform)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
        # 输出开始及数据集大小
        print(f"Start Finetune!  len_dataset: {data_set.__len__()}")

        model = NetStage2().to(device).train()
        model.load_state_dict(torch.load(self.load_model_path))  # 模型参数加载
        criterion = nn.MSELoss()
        current_lr = args.lr  # 步长
        # optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, betas=(0.9, 0.999))
        optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum)  # 优化器

        collect_loss = [['epoch', 'current_lr', 'epoch_mean_loss', 'epoch_max_loss', 'epoch_min_loss',
                         'test_mean_loss', 'test_max_loss', 'test_min_loss']]
        epoch_count = []
        loss_record = []
        cost_time_record = []
        for i in range(args.epochs):  # 开始训练
            epoch_loss = []  # 每个epoch的loss和
            for index, (img, label) in enumerate(data_loader):
                img = img.to(device)  # 数据输入设备
                label = torch.from_numpy(label.numpy().astype('float32')).to(device)  # 标签输入设备
                optimizer.zero_grad()  # 优化器梯度清零
                output = model(img)  # 前向传播计算模型输出
                loss = criterion(output, label)  # 计算loss值
                epoch_loss.append(loss.item())  # 采集epoch内的loss
                loss.backward()  # loss值对模型内参数进行反向传播
                optimizer.step()  # 优化器更新模型参数
            if i % 100 == 0:
                self.show_face_landmarks(img, i, labels=label, outputs=output)
            epoch_mean_loss = sum(epoch_loss) / (len(epoch_loss))
            epoch_max_loss = max(epoch_loss)
            epoch_min_loss = min(epoch_loss)
            # Early Stop 每个epoch存储模型并检验
            mean_loss, max_loss, min_loss = self.test(model.state_dict())
            # 供visdom显示
            epoch_count.append(i + 1)
            loss_record.append([epoch_mean_loss, epoch_max_loss, epoch_min_loss, mean_loss, max_loss, min_loss])
            cost_time_record.append(time.time() - self.start_time)
            vis.line(X=epoch_count, Y=loss_record, win='chart1', opts=opts1)
            vis.line(X=epoch_count, Y=cost_time_record, win='chart2', opts=opts2)
            
            # 采集epoch序数和每个patch的平均、最大、最小loss
            collect_loss.append([i, current_lr, epoch_mean_loss, epoch_max_loss, epoch_min_loss,
                                 mean_loss, max_loss, min_loss])  # 采集loss

        # 保存模型和loss
        save_model_path = self.load_model_path[:-3] + '_finetune_' + str(i) + ".pt"
        save_loss_path = self.load_model_path[:-3] + '_finetune_' + str(i) + "_loss.csv"
        torch.save(model.state_dict(), save_model_path)
        self.writelist2csv(collect_loss, save_loss_path)
        print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Finetune complete!')

    def predict(self):
        predict_result = []
        # 输出数据集大小
        print(f"Start Predict!")
        model = ResNet([2, 2, 2]).to(device)
        model.load_state_dict(torch.load(self.load_model_path))  # 模型参数加载
        model.eval()
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
            output = model(img)  # 前向传播计算模型输出
            self.show_face_landmarks([[img_ori]], output.cpu().detach().numpy(), labels=None, patch_plot_num=1)  # 查看
            predict_result.append([output.data])
            print(f'-predict_result: {output.data}')
        save_loss_path = self.load_model_path[:-3] + '_predict_result.csv'
        self.writelist2csv(predict_result, save_loss_path)
        print('Predict complete!')

    @staticmethod
    def writelist2csv(list_data, csv_name):  # 列表写入.csv
        with open(csv_name, "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for one_slice in list_data:
                csv_writer.writerow(one_slice)

    def show_face_landmarks(self, imgs, epoch, labels=None, outputs=None, plots=1):  # 绘制处理后的图片
        random_plt = random.sample(range(len(imgs)), plots)  # plots每个patch选择绘制的图片数
        for i in random_plt:
            img = imgs[i].cpu().detach().numpy().transpose(1, 2, 0)
            img = img * np.array(self.data_std) + np.array(self.data_mean)
            img = np.array(np.rint(img * 255), dtype='uint8')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)

            landmarks_l = labels[i].cpu().detach().numpy().reshape(-1, 2).T  # 预测值
            plt.scatter(landmarks_l[0], landmarks_l[1], s=3, c='r')
            if outputs is not None:
                landmarks_o = outputs[i].cpu().detach().numpy().reshape(-1, 2).T
                plt.scatter(landmarks_o[0], landmarks_o[1], s=3, c='b')
            plt.title(f'Epoch: {epoch}  Red-Label Blue-Predict')
            plt.show()

    @staticmethod
    def capture_predict_face():
        cap = cv2.VideoCapture(0)
        while True:
            _, img = cap.read()
            img_ori = img.copy()
            h, w, _ = img.shape
            c_x, c_y, rect_edge = int(w / 2), int(h / 2), int(min(h, w) * 0.3)
            cv2.rectangle(img, (c_x - rect_edge, c_y - rect_edge), (c_x + rect_edge, c_y + rect_edge), (0, 255, 0), 4)
            cv2.putText(img, 'Please put your face in the green rectangle.',
                        (int(h * 0.0), int(w * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 4)
            cv2.putText(img, "'Enter' to go on predict, 'Esc' to quit.",
                        (int(h * 0.08), int(w * 0.12)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 4)
            cv2.imshow("img", img)
            k = cv2.waitKey(1)
            if k == 27:  # Esc
                img = []
                break
            elif k == 13:  # Enter
                capture_img = img_ori[c_y - rect_edge: c_y + rect_edge, c_x - rect_edge: c_x + rect_edge, :]
                img = cv2.resize(capture_img, (224, 224), interpolation=cv2.INTER_AREA)
                break
        cv2.destroyAllWindows(), cap.release()
        return img, k


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--train-set-ratio', type=float, default=0.8, metavar='N',
                        help='train percentage of all data (default: 0.8)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0000001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--save-directory', type=str, default='save_stage2_model',
                        help='learnt models are saving here')
    parser.add_argument('--re-generate-data', type=bool, default=False,
                        help='if need to re generate data')
    parser.add_argument('--re-cal-mean-std', type=bool, default=False,
                        help='if need to re calculate dateset mean and std')
    parser.add_argument('--work', type=str, default='finetune',  # train, test, finetune, predict
                        help='training, test, predict or finetune')
    args = parser.parse_args()

    # visdom可视化设置
    vis = Visdom(env="face key point detection stage Dropout and BN and Plus FC")
    assert vis.check_connection()
    opts1 = {
        "title": 'loss of mean/max/min in epoch',
        "xlabel": 'epoch',
        "ylabel": 'loss',
        "width": 600,
        "height": 400,
        "legend": ['train_mean_loss', 'train_max_loss', 'train_min_loss', 'test_mean_loss', 'test_max_loss',
                   'test_min_loss']
    }
    opts2 = {
        "title": 'cost time with epoch',
        "xlabel": 'epoch',
        "ylabel": 'time in second',
        "width": 400,
        "height": 300,
        "legend": ['cost time']
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_1 = torch.device("cpu")
    assignment = MultiWorks(load_model_path='save_stage2_model/202008020752_train_epoch_499_finetune_499_finetune_499_finetune_499_finetune_499_finetune_499_finetune_499_finetune_499_finetune_499.pt')
    # 202006262114_train_epoch_999.pt
