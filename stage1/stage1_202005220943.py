# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils import data
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import time
import os
import random
from visdom import Visdom
import csv
import shutil
import collections
import os


class GenerateImageData:
    def __init__(self):
        self.data_path = ['data/I/', 'data/II/']  # 数据位置
        self.expand_ratio = 0.25  # 人脸框扩张率
        self.trainset_ratio = 0.8  # 训练集占比
        self.boundary_size = 112  # 裁剪图片尺寸
        self.shuffle_seed = 42  # 数据集洗牌随机种子
        self.data_save_path = ['stage1_train_data/', 'stage1_test_data/']  # 数据位置
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

        file = open(self.data_save_path[0] + 'label.txt', 'w')  # 生成训练集
        name_list = []
        for _i in self.train_set:
            name = _i['img_path'].split("/", 2)[-1]
            name_list.append(name)
            name_count = collections.Counter(name_list)
            name_times = name_count[name]
            name_split = name.split('.')
            img_save_path = self.data_save_path[0] + name_split[0] + '_' + str(name_times) + '.' + name_split[1]

            img_crop, relative_landmarks = self.expand_crop(_i)  # 扩大人脸框并剪裁
            img, label = self.proportion_resize(img_crop, relative_landmarks)  # 保持x y轴比例进行resize
            str_path_label = img_save_path
            for _j in label:
                str_path_label = str_path_label + ' ' + str(_j)
            file.write(str_path_label + chr(10))
            cv2.imwrite(img_save_path, img)
        file.close()

        file = open(self.data_save_path[1] + 'label.txt', 'w')  # 生成测试集
        name_list = []
        for _i in self.test_set:
            name = _i['img_path'].split("/", 2)[-1]
            name_list.append(name)
            name_count = collections.Counter(name_list)
            name_times = name_count[name]
            name_split = name.split('.')
            img_save_path = self.data_save_path[1] + name_split[0] + '_' + str(name_times) + '.' + name_split[1]

            img_crop, relative_landmarks = self.expand_crop(_i)  # 扩大人脸框并剪裁
            img, label = self.proportion_resize(img_crop, relative_landmarks)  # 保持x y轴比例进行resize
            str_path_label = img_save_path
            for _j in label:
                str_path_label = str_path_label + ' ' + str(_j)
            file.write(str_path_label + chr(10))
            cv2.imwrite(img_save_path, img)
        file.close()

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
    def __init__(self, transform=None):
        generate_data = GenerateImageData()
        self.data_save_path = generate_data.data_save_path  # 数据位置
        self.transform = transform  # 输入数据转换
        train_set, test_set = self.read_txt()  # 调用读取txt函数，得到数据字典
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
        mean = [imgs[:, :, _j, :].mean() for _j in range(3)]
        std = [imgs[:, :, _j, :].std() for _j in range(3)]
        return mean, std

    def __getitem__(self, idx):  # 返回单个样本的数据和标签，供DataLoader调用
        img_path, label = self.data_set[idx].values()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.transform is not None:  # 若有transform操作则进行transform
            img = self.transform(img)
        return img, label

    def __len__(self):  # 返回数据集大小
        return len(self.data_set)


class NetStage1(nn.Module):  # 模型定义
    def __init__(self):
        super(NetStage1, self).__init__()

        block1 = nn.Sequential()
        block1.add_module('conv1_1', nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=0))  # single_img: 8 * 54 * 54
        block1.add_module('prelu_conv1_1', nn.PReLU())
        block1.add_module('pool1', nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))  # single_img: 8 * 27 * 27
        self.block1 = block1

        block2 = nn.Sequential()
        block2.add_module('conv2_1', nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0))  # single_img: 16 * 25 * 25
        block2.add_module('prelu_conv2_1', nn.PReLU())
        block2.add_module('conv2_2', nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0))  # single_img: 16 * 23 * 23
        block2.add_module('prelu_conv2_2', nn.PReLU())
        block2.add_module('pool2',
                          nn.AvgPool2d(kernel_size=2, stride=2, padding=1, ceil_mode=True))  # single_img: 16 * 12 * 12
        self.block2 = block2

        block3 = nn.Sequential()
        block3.add_module('conv3_1', nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=0))  # single_img: 24 * 10 * 10
        block3.add_module('prelu_conv3_1', nn.PReLU())
        block3.add_module('conv3_2', nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=0))  # single_img: 24 * 8 * 8
        block3.add_module('prelu_conv3_2', nn.PReLU())
        block3.add_module('pool3', nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))  # single_img: 24 * 4 * 4
        self.block3 = block3

        block4 = nn.Sequential()
        block4.add_module('conv4_1', nn.Conv2d(24, 40, kernel_size=3, stride=1, padding=1))  # single_img: 40 * 4 * 4
        block4.add_module('prelu_conv4_1', nn.PReLU())
        block4.add_module('conv4_2', nn.Conv2d(40, 80, kernel_size=3, stride=1, padding=1))  # single_img: 80 * 4 * 4
        block4.add_module('prelu_conv4_2', nn.PReLU())
        self.block4 = block4

        block5 = nn.Sequential()
        block5.add_module('ip1', nn.Linear(1280, 128))
        block5.add_module('prelu_ip1', nn.PReLU())
        block5.add_module('ip2', nn.Linear(128, 128))
        block5.add_module('prelu_ip2', nn.PReLU())
        block5.add_module('landmarks', nn.Linear(128, 42))
        self.block5 = block5

    def forward(self, x):  # 前向传播
        result_block1 = self.block1(x)
        result_block2 = self.block2(result_block1)
        result_block3 = self.block3(result_block2)
        result_block4 = self.block4(result_block3)
        fc_input = result_block4.view(result_block4.size(0), -1)  # 进行压平操作
        result_block5 = self.block5(fc_input)
        return result_block5


class MultiWorks:  # 多任务
    def __init__(self, load_model_path=None):
        self.load_model_path = load_model_path  # 微调、测试和预测时提供模型加载路径
        # 数据集
        self.data_mean, self.data_std = DataSet().data_mean, DataSet().data_std
        self.data_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(tuple(self.data_mean), tuple(self.data_std))])
        self.data_set = DataSet(transform=self.data_transform)

        # 选择任务
        if args.work not in ['train', 'test', 'finetune', 'predict']:
            print("The args.work should be one of ['train', 'test', 'finetune', 'predict']")
        elif args.work == "train":  # 训练
            self.train()
        elif self.load_model_path is None:
            print("Please input 'load_model_path'")
        elif args.work == "test":  # 验证
            self.test()
        elif args.work == "finetune":  # 调模型
            self.finetune()
        elif args.work == "predict":  # 预测
            self.predict()

    def train(self):
        start_time = time.time()  # 开始训练时间，用于输出训练时长
        data_loader = torch.utils.data.DataLoader(self.data_set, batch_size=args.batch_size, shuffle=True)
        # 输出开始及数据集大小
        print(f"Start Train!  len_dataset: {self.data_set.__len__()}")

        model = NetStage1().to(device)
        for m in model.modules():  # 参数初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)  # 卷积层kaiming初始化
                m.bias.data.fill_(0)  # 偏差初始为零
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_()  # 全连接层随机初始化

        criterion = nn.MSELoss()  # 损失函数
        current_lr = args.lr  # 步长
        optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum * 0)  # 优化器

        # 采集loss存储为.csv
        collect_loss = [['epoch', 'current_lr', 'epoch_mean_loss', 'epoch_max_loss', 'epoch_min_loss']]
        epoch_count = []
        loss_record = []
        cost_time_record = []
        for i in range(args.epochs):  # 开始训练
            epoch_loss = []  # 每个epoch的loss和
            for index, (img, label) in enumerate(data_loader):
                img = img.to(device)  # 图片输入设备
                label = torch.from_numpy(label.numpy().astype('float32')).to(device)  # 标签输入设备
                optimizer.zero_grad()  # 优化器梯度清零
                output = model(img)  # 前向传播计算模型输出
                loss = criterion(output, label)  # 计算loss值
                epoch_loss.append(loss.item())  # 采集epoch内的loss
                loss.backward()  # loss值对模型内参数进行反向传播
                optimizer.step()  # 优化器更新模型参数
            if i % 20 == 0:
                self.show_face_landmarks(img, i, labels=label, outputs=output)
            epoch_mean_loss = sum(epoch_loss) / (len(epoch_loss))
            epoch_max_loss = max(epoch_loss)
            epoch_min_loss = min(epoch_loss)
            # 供visdom显示
            epoch_count.append(i + 1)
            loss_record.append([epoch_mean_loss, epoch_max_loss, epoch_min_loss])
            cost_time_record.append(time.time() - start_time)
            vis.line(X=epoch_count, Y=loss_record, win='chart1', opts=opts1)
            vis.line(X=epoch_count, Y=cost_time_record, win='chart2', opts=opts2)
            # 采集epoch序数和每个patch的平均、最大、最小loss
            collect_loss.append([i, current_lr, epoch_mean_loss, epoch_max_loss, epoch_min_loss])

            if i == 5:
                current_lr = args.lr * 100
                optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum)  # 优化函数
            if i == 15:
                current_lr = args.lr * 500
                optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum)  # 优化函数

        # 保存模型和loss
        if args.save_model:  # 是否保存模型
            if not os.path.exists(args.save_directory):  # 新建保存文件夹
                os.makedirs(args.save_directory)
            save_model_path = os.path.join(args.save_directory,
                                           time.strftime('%Y%m%d%H%M') + '_train_epoch_' + str(i) + ".pt")
            save_loss_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M') + '_train_loss.csv')
            torch.save(model.state_dict(), save_model_path)
            self.writelist2csv(collect_loss, save_loss_path)
            print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Train complete!')

    def test(self):
        data_loader = torch.utils.data.DataLoader(self.data_set, batch_size=args.test_batch_size, shuffle=False)
        # 输出数据集大小
        print(f"Start Test!  len_dataset: {self.data_set.__len__()}")
        model = NetStage1().to(device)
        model.load_state_dict(torch.load(self.load_model_path))  # 模型参数加载
        model.eval()  # 关闭参数梯度
        criterion = nn.MSELoss()  # 损失函数

        collect_test_loss = [['mean_patch_loss', 'max_patch_loss', 'min_patch_loss']]
        patch_loss = []  # test_set的loss
        for index, (img, label) in enumerate(data_loader):
            img = img.to(device)  # 图片输入设备
            label = torch.from_numpy(label.numpy().astype('float32')).to(device)  # 标签输入设备
            output = model(img)  # 前向传播计算模型输出
            loss = criterion(output, label)  # 计算loss值
            patch_loss.append(loss.item())
            # 查看模型输出及标签
            self.show_face_landmarks(img, 1, labels=label, outputs=output)
        mean_patch_loss = sum(patch_loss) / len(patch_loss)
        max_patch_loss = max(patch_loss)
        min_patch_loss = min(patch_loss)
        print(
            f'mean_patch_loss: {mean_patch_loss}  max_batch_loss: {max_patch_loss}  min_batch_loss: {min_patch_loss}')
        collect_test_loss.append([mean_patch_loss, max_patch_loss, min_patch_loss])
        save_loss_path = self.load_model_path[:-3] + '_test_result.csv'
        self.writelist2csv(collect_test_loss, save_loss_path)
        print(f'--Save complete!\n--save_loss_path: {save_loss_path}\n')
        print('Eval complete!')

    def finetune(self):
        start_time = time.time()  # 开始训练时间，用于输出训练时长
        data_loader = torch.utils.data.DataLoader(self.data_set, batch_size=args.batch_size, shuffle=True)
        # 输出数据集大小
        print(f"Start Finetune!  len_dataset: {self.data_set.__len__()}")
        model = NetStage1().to(device)
        model.load_state_dict(torch.load(self.load_model_path))

        criterion = nn.MSELoss(reduction='sum')  # 损失函数
        current_lr = args.lr * 0.1  # 步长
        optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum)  # 优化器

        # 采集loss存储为.csv
        collect_loss = [['epoch', 'current_lr', 'epoch_mean_loss', 'epoch_max_loss', 'epoch_min_loss']]
        epoch_count = []
        loss_record = []
        cost_time_record = []
        for i in range(args.epochs):  # 开始训练
            epoch_loss = []  # 每个epoch的loss和
            for index, (img, label) in enumerate(data_loader):
                img = img.to(device)  # 图片输入设备
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
            # 供visdom显示
            epoch_count.append(i + 1)
            loss_record.append([epoch_mean_loss, epoch_max_loss, epoch_min_loss])
            cost_time_record.append(time.time() - start_time)
            vis.line(X=epoch_count, Y=loss_record, win='chart1', opts=opts1)
            vis.line(X=epoch_count, Y=cost_time_record, win='chart2', opts=opts2)
            # 采集epoch序数和每个patch的平均、最大、最小loss
            collect_loss.append([i, current_lr, epoch_mean_loss, epoch_max_loss, epoch_min_loss])

        # 保存模型和loss
        save_finetune_model_path = self.load_model_path[:-3] + '_finetune_' + str(i) + ".pt"
        save_finetune_loss_path = self.load_model_path[:-3] + '_finetune_' + str(i) + "_loss.csv"
        torch.save(model.state_dict(), save_finetune_model_path)
        self.writelist2csv(collect_loss, save_finetune_loss_path)
        print(f'--Save complete!\n--save_finetune_model_path: {save_finetune_model_path}\n'
              f'--save_finetune_loss_path: {save_finetune_loss_path}')
        print('Finetune complete!')

    def predict(self):
        predict_result = [['img_save_path, predict_face_landmark']]
        predict_save_path = 'predict_stage1/'
        if not os.path.exists(predict_save_path):  # 新建保存文件夹
            os.makedirs(predict_save_path)
        data_loader = torch.utils.data.DataLoader(self.data_set, batch_size=1, shuffle=True)


        # 输出数据集大小
        print(f"Start Predict!")
        model = NetStage1().to(device)
        model.load_state_dict(torch.load(self.load_model_path))
        model.eval()

        while True:
            img_ori, k = self.capture_predict_face()
            if k == 27:
                print('Quit Predict!')
                break

            img_show = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            img_show_compare = self.data_set.__getitem__(np.random.randint(0, 500))[0].unsqueeze(0).to(device)
            # plt.imshow(img_show)
            # plt.show()
            # print(img_show)
            # print()
            img_transform = self.data_transform(img_ori)
            # img_transform = transforms.ToTensor()(img_show)
            img = img_transform.unsqueeze(0).to(device)  # 数据输入设备
            plt.subplot(121)
            img_compare = img_show_compare[0].cpu().detach().numpy().transpose(1, 2, 0) * self.data_std + self.data_mean
            plt.imshow(img_compare)

            print('sample')
            print(np.mean(img_show_compare[0].cpu().detach().numpy().transpose(1, 2, 0)),
                  np.std(img_show_compare[0].cpu().detach().numpy().transpose(1, 2, 0)),
                  np.max(img_show_compare[0].cpu().detach().numpy().transpose(1, 2, 0)),
                  np.min(img_show_compare[0].cpu().detach().numpy().transpose(1, 2, 0)))
            plt.title('sample')
            plt.subplot(122)
            plt.imshow(img_ori)
            print('real')
            print(np.mean(img[0].cpu().detach().numpy().transpose(1, 2, 0)),
                  np.std(img[0].cpu().detach().numpy().transpose(1, 2, 0)),
                  np.max(img[0].cpu().detach().numpy().transpose(1, 2, 0)),
                  np.min(img[0].cpu().detach().numpy().transpose(1, 2, 0)))
            plt.title('real')
            plt.show()
            for i, (image, label) in enumerate(data_loader):
                plt.scatter(np.mean(image.numpy()), np.std(image.numpy()), s=1, c='b')
            plt.scatter(np.mean(img_show_compare[0].cpu().detach().numpy().transpose(1, 2, 0)),
                        np.std(img_show_compare[0].cpu().detach().numpy().transpose(1, 2, 0)), s=10, c='k')
            plt.scatter(np.mean(img[0].cpu().detach().numpy().transpose(1, 2, 0)),
                        np.std(img[0].cpu().detach().numpy().transpose(1, 2, 0)), s=10, c='r')
            plt.title('mean - std')
            plt.show()
            for i, (image, label) in enumerate(data_loader):
                plt.scatter(np.max(image.numpy()), np.min(image.numpy()), s=1, c='b')
            plt.scatter(np.max(img_show_compare[0].cpu().detach().numpy().transpose(1, 2, 0)),
                        np.min(img_show_compare[0].cpu().detach().numpy().transpose(1, 2, 0)), s=10, c='k')
            plt.scatter(np.max(img[0].cpu().detach().numpy().transpose(1, 2, 0)),
                        np.min(img[0].cpu().detach().numpy().transpose(1, 2, 0)), s=10, c='r')
            plt.title('max - min')
            plt.show()
            output = model(img)  # 前向传播计算模型输出
            self.show_face_landmarks(img, 0, outputs=output)  # 查看预测结果
            output1 = model(img_show_compare)  # 前向传播计算模型输出
            self.show_face_landmarks(img_show_compare, 0, outputs=output1)  # 查看预测结果

            save_img_path = os.path.join(predict_save_path, time.strftime('%Y%m%d%H%M') + ".jpg")
            cv2.imwrite(save_img_path, img_ori)
            predict_result.append([save_img_path, output[0].data])
        save_loss_path = self.load_model_path[:-3] + '_predict_result.csv'
        self.writelist2csv(predict_result, save_loss_path)
        print('Predict complete!')

    @staticmethod
    def writelist2csv(list_data, csv_name):  # 列表写入.csv
        with open(csv_name, "w", newline='') as csvFile:
            csvWriter = csv.writer(csvFile)
            for data in list_data:
                csvWriter.writerow(data)

    def show_face_landmarks(self, imgs, epoch, labels=None, outputs=None, plots=1):  # 绘制处理后的图片
        random_plt = random.sample(range(len(imgs)), plots)  # plots每个patch选择绘制的图片数
        for i in random_plt:
            img = imgs[i].cpu().detach().numpy().transpose(1, 2, 0)
            img = img * np.array(self.data_std) + np.array(self.data_mean)
            img = np.array(np.rint(img * 255), dtype='uint8')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)

            if labels is not None:
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
                img = cv2.resize(capture_img, (96, 96))
                img = cv2.resize(img, (112, 112))
                break
        cv2.destroyAllWindows(), cap.release()
        return img, k


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--trainset-ratio', type=float, default=0.8, metavar='N',
                        help='train percentage of all data (default: 0.8)')
    parser.add_argument('--batch-size', type=int, default=2152, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.00000000001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='save_stage1_model',
                        help='learnt models are saving here')
    parser.add_argument('--re-generate-data', type=bool, default=False,
                        help='if need to re generate data')
    parser.add_argument('--re-cal-mean-std', type=bool, default=False,
                        help='if need to re calculate dateset mean and std')
    parser.add_argument('--work', type=str, default='test',  # train, test, finetune, predict
                        help='train, test, finetune, predict')
    args = parser.parse_args()

    # visdom可视化设置
    vis = Visdom(env="face key point detection stage1")
    assert vis.check_connection()
    opts1 = {
        "title": 'loss of mean/max/min in epoch',
        "xlabel": 'epoch',
        "ylabel": 'loss',
        "width": 600,
        "height": 400,
        "legend": ['mean_loss', 'max_loss', 'min_loss']
    }
    opts2 = {
        "title": 'cost time with epoch',
        "xlabel": 'epoch',
        "ylabel": 'time in second',
        "width": 400,
        "height": 300,
        "legend": ['cost time']
    }

    # 任务开始
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assignment = MultiWorks(
        load_model_path='save_stage1_model/202005231913_train_epoch_99_finetune_99_finetune_299_finetune_999.pt')
