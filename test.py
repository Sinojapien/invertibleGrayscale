import torch
from torch import nn
import torch.nn.functional as func

import numpy as np

import torchvision.transforms
from torchvision import datasets, transforms
from imports.JPEG.DiffJPEG import DiffJPEG
import data

from imports.pytorch_ssim import ssim


def mean(items):
    return np.array(items).mean()


def stddev(items):
    return np.array(items).std()


print(mean([1.5, 2.5, 3.5, 5.5]))
print(stddev([1.5, 2.5, 3.5, 5.5]))
exit()


def tensor_to_numpy(image : torch.Tensor):
    return image.detach().numpy()


def SSIM_difference(image, target):
    # https://ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python
    # https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html
    # https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
    # https://blog.csdn.net/Mao_Jonah/article/details/117228926
    # https://github.com/scikit-image/scikit-image/issues/4636
    # https://github.com/Po-Hsun-Su/pytorch-ssim
    # https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html
    return ssim(image.detach(), target.detach())


sample = torch.rand(4, 3, 256, 256, requires_grad=True)
random = torch.rand(4, 3, 256, 256, requires_grad=True)
print(type(tensor_to_numpy(sample)))
print(tensor_to_numpy(sample).shape)
print(tensor_to_numpy(sample).max())
print(tensor_to_numpy(sample).min())
print(SSIM_difference(sample, random).item())
exit()

alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
print(alpha)
exit()

stage_three = True
current_epochs = 120
max_epochs = 120
num_of_epochs = 30

# def lr_lambda(epoch):
#     if not stage_three:
#         output = 1 - 0.99 * current_epochs / max_epochs
#     else:
#         output = 1 - 0.99 * epoch / num_of_epochs
#     if output < 0.01:
#         output = 0.01
#     elif output > 1:
#         output = 1.0
#     return output
#
# for i in range(30):
#     print(lr_lambda(i))
#     current_epochs += 1
# exit()


jpeg = DiffJPEG(height=16, width=16, differentiable=True, quality=data.JPEG_quality)
def apply_dct_artifact(image, grayscale_input=False, quantize=True):
    image = (image + 1) / 2
    if grayscale_input:
        image = image.expand(-1, 3, -1, -1)
    image = jpeg(image)
    if grayscale_input:
        image = torch.unsqueeze(torch.mean(image, dim=1), dim=1)
    image = image * 2 - 1
    if quantize:
        image = data.quantize(image, use_round=False)
    return image

sample = torch.ones(1, 3, 16, 16, requires_grad=True)
sample_jpeg = apply_dct_artifact(sample * 0.5, grayscale_input=True, quantize=True)
torch.nn.L1Loss()(sample_jpeg, torch.ones_like(sample_jpeg) * 0.5).backward()
print(sample_jpeg.size())
print(sample.grad)

exit()


def deSample(image, scale):
    if scale <= 0.0: scale = 1.0
    forward_value = nn.functional.interpolate(image, scale_factor=1.0/scale, mode='nearest')
    forward_value = nn.functional.interpolate(forward_value, scale_factor=scale, mode='nearest')
    output = image.clone()
    output.data = forward_value.data
    return output
#     if scale <= 0.0:
#         return image
#     intermediate = nn.functional.interpolate(image, scale_factor=1.0 / scale, recompute_scale_factor=True)
#     return nn.functional.interpolate(intermediate, scale_factor=scale, recompute_scale_factor=True)


# sample = torch.randn(1, 3, 256, 256, requires_grad=True)
# d_sample = deSample(sample, 4.0)
# print(sample)
# print(d_sample)
#
# torch.nn.L1Loss()(d_sample, torch.ones_like(d_sample)).backward()
# print(sample.grad)
#
# exit()


jpeg = DiffJPEG(height=data.input_resolution, width=data.input_resolution, differentiable=True, quality=data.JPEG_quality)
grayscale_transform = torchvision.transforms.Grayscale()
conv = torch.nn.Conv2d(3, 3, 3, 1, 1)
# https://discuss.pytorch.org/t/torch-round-gradient/28628/5
# https://github.com/kitayama1234/Pytorch-BPDA
def floor_func_BPDA(input):
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    forward_value = torch.floor(input)
    out = input.clone()
    out.data = forward_value.data
    return out

def apply_dct_artifact(image):
    # https://asecuritysite.com/comms/dct
    # https://www.math.cuhk.edu.hk/~lmlui/J4DCT-Huff2009.pdf
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1377358&tag=1
    # https://github.com/zh217/torch-dct
    # https://github.com/pytorch/pytorch/issues/49016
    # https://queuecumber.gitlab.io/torchjpeg/api/torchjpeg.dct.html
    # https://pytorch.org/audio/stable/functional.html?highlight=dct#torchaudio.functional.create_dct
    # https://github.com/wbhu/Mono3D/blob/master/models/CNS/jpg_module.py
    # http://www.cse.cuhk.edu.hk/~ttwong/papers/invbino/invbino.pdf
    # Compress
    # Uncompress
    # input [-1, 1], jpeg [0, 1], output [-1, 1]
    image = (image + 1) / 2
    image = image.expand(-1, 3, -1, -1)
    image = jpeg(image)
    image = torch.unsqueeze(torch.mean(image, dim=1), dim=1)
    image = image * 2 - 1
    return image

sample = torch.randn(1, 3, 256, 256, requires_grad=True)
conv_sample = conv(sample)
# quant_sample_jpeg = apply_dct_artifact(data.quantize(sample))
sample_jpeg = apply_dct_artifact(conv_sample)
# print(sample_jpeg)
# print(data.quantize(quant_sample_jpeg))
# print(data.quantize(sample_jpeg))

# quant = data.quantize(conv_sample)
# print(quant)
# no_grad_quant = torch.floor((conv_sample + 1) * 127.5)
# print(no_grad_quant / 127.5 - 1)

torch.nn.L1Loss()(sample_jpeg, torch.ones_like(sample_jpeg)).backward()
print(sample.grad)
print(conv.weight.grad)

exit()

print(np.random.rand())
# print(np.random.randint(101))
exit()

# moreTensor_A = torch.rand((2, 1, 2, 2), requires_grad=True)
# moreTensor_A_stack = moreTensor_A.expand(-1, 3, -1, -1)
# # print(moreTensor_A[:, 0, :, :].size())
# # print(torch.squeeze(moreTensor_A, dim=1).size())
# print(moreTensor_A)
# print(torch.unsqueeze(torch.mean(moreTensor_A, dim=1), dim=1))
# exit()

# from imports import JPEGCompression as jpeg
#
# moreTensor_A = torch.rand((2, 3, 2, 2), requires_grad=True)
# X = jpeg.dct_2d(moreTensor_A)
# Y = jpeg.idct_2d(X)
# print(moreTensor_A)
# print(Y)
# Y.backward()
# print(moreTensor_A.grad)
# exit()

from imports.JPEG.DiffJPEG import DiffJPEG
jpeg = DiffJPEG(height=16, width=16, differentiable=True, quality=80)

moreTensor_A = torch.rand((2, 1, 16, 16), requires_grad=True)
# moreTensor_A = torch.ones((2, 1, 16, 16), requires_grad=True)
moreTensor_A_processed = moreTensor_A * 2 - 1
print(moreTensor_A_processed)
moreTensor_A_processed = moreTensor_A_processed.expand(-1, 3, -1, -1)
moreTensor_A_processed = jpeg(moreTensor_A_processed)
print(moreTensor_A_processed)
torch.mean(moreTensor_A_processed).backward()
print(moreTensor_A.grad)
exit()

# moreTensor_A = torch.rand((3, 2, 2), requires_grad=True)
# print(moreTensor_A)
# image_A = torchvision.transforms.ToPILImage()(moreTensor_A).convert("RGB")
# Tensor_A = torchvision.transforms.ToTensor()(image_A)
# print(Tensor_A)
# print(torch.floor((moreTensor_A * 2) * 127.5) / 127.5 / 2)
# exit()


# import architecture
# import data as data

# moreTensor_A = torch.rand((2, 1, 2, 2), requires_grad=True)
# moreTensor_B = torch.rand((2, 1, 2, 2), requires_grad=True)
# moreTensor_B_processed = moreTensor_B * 0.25 + 1
# loss = torch.nn.functional.l1_loss(moreTensor_A, moreTensor_B_processed, reduction='mean')
# # loss = torch.nn.functional.l1_loss(moreTensor_B, moreTensor_A, reduction='mean')
# loss.backward()
# print(moreTensor_A.grad)
# print(moreTensor_B.grad)
# exit()

# moreTensor_A = torch.rand((2, 1, 2, 2), requires_grad=True)
moreTensor_B = torch.rand((2, 1, 2, 2), requires_grad=True)
moreTensor_C = torch.rand((2, 3, 2, 2), requires_grad=True)
moreTensor_A = torch.ones((2, 1, 2, 2), requires_grad=True)
# moreTensor_A_normalized = (moreTensor_A - 0.5) * 2
# moreTensor_B_normalized = (moreTensor_B - 0.5) * 2
# moreTensor_C_normalized = (moreTensor_C - 0.5) * 2
print(moreTensor_A)

combined = torch.concat([moreTensor_A for t in range(256)], dim=0)
# a = torch.ones_like(moreTensor_A.detach())
a = torch.ones((2, 1, 1, 1))
combinedOnes = torch.concat([a * t for t in range(256)], dim=0)
combinedOnes = (combinedOnes / 127.5) - 1
print(combined)
print(combinedOnes)
print(combined.size())
print(combinedOnes.size())
# minTensorA = torch.min(torch.min(torch.abs(combined - combinedOnes), dim=3).values, dim=2).values
minTensorA = torch.min(torch.abs(combined - combinedOnes), dim=0).values
# minTensorB = torch.min(torch.min(torch.abs(combined - combinedOnes), dim=2).values, dim=2).values
print(minTensorA)
print(minTensorA.size())
sum_loss = torch.mean(minTensorA)
print(sum_loss)
# print(torch.max(minTensorA - minTensorB))
# loss = torch.nn.functional.l1_loss(combined, combinedOnes)
# print(loss)
# loss.backward()
sum_loss.backward()
print(moreTensor_A.grad)

# moreTensor_B_target = (torch.round((moreTensor_B.detach() + 1) * 127.5) / 127.5 - 1)
# diff_1 = torch.nn.functional.l1_loss(moreTensor_B, moreTensor_B_target, reduction='mean')
# # diff_1 = torch.mean(torch.abs(moreTensor_B - moreTensor_B_target))
# diff_1.backward()
# print(diff_1)
# print(moreTensor_B.grad)

# gradient are static
# grayscale_C = torchvision.transforms.Grayscale()
# loss_C = torch.nn.functional.l1_loss(moreTensor_C, torch.rand((2, 3, 2, 2)))
# loss_C.backward()
# print(moreTensor_C.grad)

exit()


moreTensor_A = (torch.rand((2, 1, 2, 2)) - 0.5) * 2
moreTensor_B = moreTensor_A.detach()
moreTensor_C = moreTensor_A.detach()
moreTensor_A.requires_grad = True
moreTensor_B.requires_grad = True
moreTensor_C.requires_grad = True
# print(moreTensor_A)

normalized = (moreTensor_A + 1) * 127.5
normalized_round = torch.round(normalized.detach())
normalized_int = normalized.type(torch.IntTensor)
# print(moreTensor_A)
# print(normalized)
# print(normalized_int)
# print(normalized_round)
diff = torch.nn.functional.l1_loss(normalized, normalized_round)
diff.backward()
moreTensor_B_target = (torch.round((moreTensor_B.detach() + 1) * 127.5) / 127.5 - 1).detach()
print(moreTensor_B - moreTensor_B_target)
diff_1 = torch.nn.functional.l1_loss(moreTensor_B, moreTensor_B_target, reduction='mean')
# Wrong as [-127.5, 127.5] has wrong round up (bound is not integer)
diff_2 = torch.nn.functional.l1_loss(moreTensor_C, torch.round(moreTensor_C.detach() * 127.5) / 127.5)
diff_1.backward()
diff_2.backward()
# print(diff)
print(diff_1)
# print(diff_2)
# print(moreTensor_A.grad)
print(moreTensor_B.grad)
# print(moreTensor_C.grad)
exit()


def get_total_variation_a(image):
    variation_x = image[:, :, 1:, :] - image[:, :, :-1, :]
    variation_y = image[:, :, :, 1:] - image[:, :, :, :-1]

    reduce_axes = (-3, -2, -1)
    sum_x = variation_x.abs().sum(dim=reduce_axes)
    sum_y = variation_y.abs().sum(dim=reduce_axes)

    return sum_x + sum_y

def get_total_variation_b(image):
    batch_size, num_of_channels, width, height = image.size()
    variation_x = torch.pow(image[:, :, 1:, :] - image[:, :, :-1, :], 2).sum()
    variation_y = torch.pow(image[:, :, :, 1:] - image[:, :, :, :-1], 2).sum()

    return (variation_x + variation_y)

def get_total_variation_c(image):
    batch_size, num_of_channels, width, height = image.size()
    variation_x = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).sum()
    variation_y = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).sum()

    return (variation_x + variation_y) / (batch_size * num_of_channels * width * height)

tensorA = torch.randn((8, 3, 4, 4), requires_grad=True)
tensorR = torch.ones((8, 3, 4, 4))
tensorB = torch.ones((8, 3, 4, 4))
print(get_total_variation_a(tensorA))
print(get_total_variation_b(tensorA))
print(get_total_variation_c(tensorA))
# batchsize, num_of_channels, width, height = tensorA.size()
# print((batchsize, num_of_channels, width, height))
# print(tensorA[:, :, 1:, :].size())
# print(tensorA[:, :, :-1, :].size())
exit()

print(torchvision.__version__)
print(int(torchvision.__version__.__str__().split('.')[1]))
exit()

grayscale_tranforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    ])


def grayscale(x):
    print((x * d).size())
    return torch.sum(x * d, dim=1)


test = torch.ones((2, 3, 2, 2), requires_grad=True)
# ggg = torch.ones((2, 3, 1, 1)) * 2
# ggg = torch.ones_like(test)
# print(ggg)
# print(test * ggg)
# print((test * ggg).size())
# print(test)
print(test.size())
# abc = torch.sum(test, dim=1)
# d = torch.tensor([[0.299], [0.587], [0.114]])
d = torch.tensor([0.299, 0.587, 0.114]).unsqueeze(1).unsqueeze(1).repeat(len(test), 1, 1, 1)
# print(d)
# print(d.size())
# print(test * d)
# b = torch.ones_like(test) * 2
# print(b.size())
# print(test * b)
# abc = torch.mul(test, d)
# print(abc)
# print(abc.size())

sample = torch.randn(2, 3, 4, 4, requires_grad=True)
x = sample
grayscale_transform = torchvision.transforms.Grayscale()
normalize_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])


def normalize(x, mean=[], std=[]):
    # https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/
    mean_tensor = torch.ones_like(x)
    for dim in range(len(mean)):
        mean_tensor[0][dim] = mean_tensor[0][dim] * mean[dim]
    return x - mean_tensor


# print((grayscale_tranforms(sample[0]).unsqueeze(0) - 0.5) * 2)
print(grayscale_transform(x))
print(grayscale(x))
print((x[0][0] * 0.299 + x[0][1] * 0.587 + x[0][2] * 0.114).unsqueeze(0))
# print(normalize_transform(x))
# print(normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

# t = data.transforms.RandomCrop(size=2, pad_if_needed=True)
# print(t(sample))
# print(sample.size())
# print(sample.expand(-1, 3, -1, -1).size())
# print(sample.size())

# VGG19 = torchvision.models.vgg19(pretrained=True)



# abc = [1, 2, 3, 4, 5, 5.5, 6, 7]
# print(abc[:4])

# image_dataset = data.datasets.VOCDetection(data.paths.data_dir + '', download=True, year="2012", image_set="trainval")
# image_dataset = data.datasets.VOCSegmentation(data.paths.data_dir + '', download=True, year="2012", image_set="trainval")
# print(image_dataset.image_set)
# print(image_dataset.images)
# print(len(image_dataset))

# train_dataloader, test_dataloader = data.get_dataloader()
# print(len(train_dataloader))
# print(len(test_dataloader))
#
# train_iter = iter(train_dataloader)
#
# for i in range(len(train_dataloader)):
#     train_image, _ = next(train_iter)
#     print(train_image.size())
#
# print("done")

# sample = sample + 1
# sample.backward()
# print(sample)
