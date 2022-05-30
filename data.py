import numpy
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.data

import numpy as np

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
# import PIL

import torchvision
from torchvision import datasets, transforms

from PIL.Image import NEAREST
# from typing import Callable, Optional, Tuple, Any
import os.path

GN_G_plot_buffer = []
PN_G_plot_buffer = []
DN_G_plot_buffer = []
num_of_batches = 8
num_of_test_batches = 1
num_of_cores = 0
input_resolution = 256
train_ratio = 13758/17125
extra_transform_data = False
JPEG_quality = 70
input_dir = ''


def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'Created Directory: {path}')


class PATH:
    def __init__(self, input_path=''):
        self.path = ''
        self.output_dir = ''
        self.image_dir = ''
        self.plot_dir = ''
        self.model_dir = ''
        self.data_dir = ''
        self.test_dir = ''
        self.tensorboard_path = ''
        self.update_path(input_path)

    def update_path(self, input_path, create=False):
        self.path = str(input_path)
        self.output_dir = self.path + 'outputs/'
        self.image_dir = self.output_dir + 'images/'
        self.plot_dir = self.output_dir + 'plots/'
        self.model_dir = self.output_dir + 'model/'
        self.data_dir = self.path + 'data/'
        self.test_dir = self.data_dir + 'test/'
        self.tensorboard_path = self.output_dir + 'tensorboard/'

        if create:
            create_path(self.output_dir)
            create_path(self.image_dir)
            create_path(self.plot_dir)
            create_path(self.model_dir)
            create_path(self.tensorboard_path)
            create_path(self.data_dir)
            create_path(self.test_dir)

    def print_path(self):
        print(f'{self.output_dir} || {self.data_dir} || {self.test_dir}')


paths = PATH()
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])


class ReSizeAndCrop(object):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__(self, resolution):
        self.transform_crop = transforms.RandomCrop(size=resolution, fill=(255, 255, 255), pad_if_needed=True)
        self.transform_resize = transforms.Resize((resolution, resolution), interpolation=NEAREST)
        self.area = resolution * resolution
        self.resolution = resolution

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']
        # h, w = image.shape[:2]
        h, w = sample.size

        area = h * w
        if area > self.area:
            return self.transform_crop(sample)
        elif area < self.area:
            return self.transform_resize(sample)

        # if h < self.resolution or w < self.resolution:
        #     return self.transform_crop(sample)
        # elif h > self.resolution or w > self.resolution:
        #     return self.transform_resize(sample)

        return sample


def get_model_directory():
    return paths.model_dir


def get_tensorboard_directory():
    return paths.tensorboard_path


def update_path(input_path):
    paths.update_path(input_path, create=True)


def get_dataloader(directory='', apply_train_transform=True):
    # to-do: separate transforms for different dataset
    # data folder is loaded continuously, don't modify it when running!
    transform_list = []
    # transform_list.append(transforms.ToTensor())  # transform_list.append(transforms.ToPILImage())
    transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.RandomVerticalFlip())
    transform_list.append(transforms.RandomCrop(size=input_resolution, pad_if_needed=True))
    # transform_list.append(transforms.Resize((input_resolution, input_resolution), interpolation=NEAREST))
    if extra_transform_data:
        transform_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.0, saturation=0.1, hue=0.3))
        transform_list.append(transforms.RandomGrayscale(p=0.02))
        # transform_list.append(transforms.RandomRotation(degrees=(0, 5), fill=(255, 255, 255)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    data_transforms = transforms.Compose(transform_list)

    # https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/9
    # image_dataset = datasets.VOCDetection(paths.data_dir + directory, download=True, year="2012", image_set="trainval", transform=data_transforms)
    image_dataset = datasets.ImageFolder(directory, transform=data_transforms)
    train_size = int(len(image_dataset) * train_ratio)
    train_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [train_size, len(image_dataset) - train_size])
    test_dataset.transform = transforms.Compose([
        transforms.Resize((input_resolution, input_resolution), interpolation=NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    # train_dataset = data_transforms(train_dataset)
    # test_dataset = extra_data_tranforms(test_dataset)

    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=num_of_batches, shuffle=True,
                                                   num_workers=num_of_cores)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=num_of_test_batches, shuffle=True,
                                                   num_workers=num_of_cores)

    return train_dataloader, test_dataloader


def get_image_folder(directory='', mode='resize'):
    transform_list = []
    if mode == 'crop':
        transform_list.append(transforms.RandomCrop(size=input_resolution, pad_if_needed=True))
    elif mode == 'resize':
        transform_list.append(transforms.Resize((input_resolution, input_resolution), interpolation=NEAREST))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    data_transforms = transforms.Compose(transform_list)

    image_dataset = datasets.ImageFolder(directory, transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=True, num_workers=num_of_cores)

    return dataloader


def reSize(image, scale):
    return nn.functional.interpolate(image, scale_factor=scale)
    # Gives warning if fractional scale factor
    # return nn.functional.interpolate(input, scale_factor=scale, recompute_scale_factor=False)


def desample(image, scale):
    # https://github.com/pytorch/pytorch/issues/42772
    if scale <= 0.0: scale = 1.0
    intermediate = nn.functional.interpolate(image, scale_factor=1.0/scale, mode='nearest')
    return nn.functional.interpolate(intermediate, scale_factor=scale, mode='nearest')


def save_image(image, filename, normalize=True, directory=paths.image_dir):
    if normalize: image = image / 2 + 0.5  # activation layer tanh in range [-1, 1]
    torchvision.utils.save_image(image, directory + filename + '.png')


def plot_loss(x, y, title='', save=True):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    if save:
        plt.savefig(paths.plot_dir + title + '_plot.png')
        plt.close()
    else: plt.show()


def plot_all(suffix='', save=True):
    plot_loss(range(0, len(GN_G_plot_buffer)), GN_G_plot_buffer, 'GN_' + suffix, save)
    plot_loss(range(0, len(PN_G_plot_buffer)), PN_G_plot_buffer, 'PN_' + suffix, save)
    plot_loss(range(0, len(DN_G_plot_buffer)), DN_G_plot_buffer, 'DN_' + suffix, save)
    print('Finished Plotting')


def imshow(img, normalize=True):
    if normalize: img = img / 2 + 0.5  # unnormalize
    plt.imshow(img.squeeze().permute(1, 2, 0))  # plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()


def choose_best(input_set):
    # Choosing best output from PN
    outputs, image, min_loss = input_set[0]

    for i in range(1, len(input_set)):
        ith_outputs, ith_image, ith_loss = input_set[i]
        if ith_loss < min_loss:
            outputs = ith_outputs
            image = ith_image
            min_loss = ith_loss

    return outputs, image, min_loss


def choose_random(input_set):
    # Choose randomly
    pos = np.random.randint(len(input_set))
    return input_set[pos]


def choose_min(input_set):
    loss_64, loss_48, loss_32 = input_set
    pos = np.argmin((loss_64.item(), loss_48.item(), loss_32.item()))
    return input_set[pos]


def quantize(image, use_round=False):
    # Given range [-1, 1], [-1, 1] -> [0, 255]
    # https://github.com/kitayama1234/Pytorch-BPDA
    image = (image + 1) * 127.5
    if use_round:
        forward_value = torch.round(image)  # No gradient!!!
    else:
        forward_value = torch.floor(image)  # No gradient!!!
    image = image.clone()
    image.data = forward_value.data
    image = image / 127.5 - 1
    return image


def normalize(image, mean=0.5, std=0.5):
    return image * std + mean


def clear_cache(require_print=False):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if require_print:
            print(f'Memory allocated: {torch.cuda.memory_allocated()}')
            print(f'Memory reserved: {torch.cuda.memory_reserved()}')


def tensor_to_numpy(image : torch.Tensor) -> numpy.ndarray:
    return image.detach().permute(0, 2, 3, 1).numpy()


def get_mean(items):
    return np.array(items).mean()


def get_stddev(items):
    return np.array(items).std()
