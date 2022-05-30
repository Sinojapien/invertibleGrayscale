import torch
import torch.nn as nn
import torch.nn.functional as func

# import numpy as np
import torchvision.transforms

from imports.JPEG.DiffJPEG import DiffJPEG
import data  # import data as data
from imports.pytorch_ssim import ssim

# no activation in resblock, upsample instead of deconv

# Device Section
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device = torch.device("cuda")
new_torchvision_version = int(torchvision.__version__.__str__().split('.')[1]) >= 8
print(f'Cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Number of device(s): {torch.cuda.device_count()}')
print(f'Pytorch Version: {torch.__version__.__str__()}')
print(f'Torchvision Version > 0.8.0: {new_torchvision_version}')


class Weights:
    def __init__(self, inv, conf, quan, light, con, struct, jpeg, down):
        self.inv = inv
        self.conf = conf
        self.quan = quan

        self.light = light
        self.con = con
        self.struct = struct

        self.threshold = 70 / 255 * 2

        self.staged = False
        self.jpeg = jpeg
        self.down = down

    def reset(self):
        self.inv = 1.0
        self.conf = 1.0
        self.quan = 0.0

        self.light = 1.0
        self.con = 1e-7
        self.struct = 0.5

        self.staged = False

    def stage(self):
        self.inv = 1.0
        self.conf = 0.5
        self.quan = 10.0

        # use_round when train
        self.light = 0.5
        # self.struct = 0.1
        self.struct = 0.5 * 2
        # self.con = 1e-7 * 2

        self.jpeg = 1.0 * 3
        self.down = 1.0 * 3
        # self.threshold = 70 / 255 * 2

        self.staged = True
        self.print_info()

    def to_string(self):
        return f'i: {self.inv}, l: {self.light}, c: {self.con}, s: {self.struct}, q: {self.quan}'

    def print_info(self):
        print(self.__dict__)


loss_weights = Weights(1.0, 1.0, 0.0, 1.0, 1e-7, 0.5, 1.0, 1.0)


class LossBuffer:
    def __init__(self):
        self.encoder_loss = 0.0
        self.decoder_loss = 0.0

    def clear(self):
        self.encoder_loss = 0.0
        self.decoder_loss = 0.0

    def print_info(self):
        print(f'Encoder: {self.encoder_loss}, Decoder: {self.decoder_loss}')


def get_device():
    return device


# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L25
# https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        residual = self.layers(x)
        output = x + residual
        return output


class encodeNetwork(nn.Module):
    def __init__(self):
        super(encodeNetwork, self).__init__()

        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.downscale_128 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.downscale_256 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.res_block_sequence = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )
        self.upscale_128 = nn.Sequential(
            # UpSample instead
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upscale_64 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.downscale_128(x1)
        x3 = self.downscale_256(x2)
        x4 = self.res_block_sequence(x3)
        x5 = self.upscale_128(x4) + x2
        x6 = self.upscale_64(x5) + x1
        output = self.output_layer(x6)
        return output


class decodeNetwork(nn.Module):
    def __init__(self):
        super(decodeNetwork, self).__init__()
        ResBlockSequence = []
        ResBlockSequence.append(ResBlock(1, 64))
        for i in range(8):
            ResBlockSequence.append(ResBlock(64, 64))
        ResBlockSequence.append(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1))
        ResBlockSequence.append(nn.Tanh())
        self.layers = nn.Sequential(*ResBlockSequence)

    def forward(self, x):
        # Quantization required
        x = self.layers(x)
        return x


# Loss Section
l1Loss = nn.L1Loss()  # reduction='mean' or reduction='sum'
l2Norm = nn.MSELoss()
grayscale_convert = torch.tensor([0.299, 0.587, 0.114]).unsqueeze(1).unsqueeze(1).repeat(data.num_of_batches, 1, 1, 1).to(device)
grayscale_transform = torchvision.transforms.Grayscale()

# load pretrained VGG-19 for image classification
# get output from conv4_4
# to_be_implemented
# https://stackoverflow.com/questions/64631086/how-can-i-add-new-layers-on-pre-trained-model-with-pytorch-keras-example-given
# https://stackoverflow.com/questions/63427771/extracting-intermediate-layer-outputs-of-a-cnn-in-pytorch

# https://pytorch.org/vision/0.8/_modules/torchvision/models/vgg.html
# https://pytorch.org/hub/pytorch_vision_vgg/

# https://discuss.pytorch.org/t/can-i-get-the-middle-layers-output-if-i-use-the-sequential-module/7070


# https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg19
class CustomVGG19(nn.Module):
    def __init__(self, pretrained=False):
        super(CustomVGG19, self).__init__()
        # Stored in '~/.cache/torch/hub/checkpoints/'
        self.model = torchvision.models.vgg19(pretrained=pretrained)
        # self.layers = torchvision.models.vgg19(pretrained=False).features
        self.offset = 2*2+1 + 2*2+1 + 4*2+1 + 4*2
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def forward(self, x):
        # https://discuss.pytorch.org/t/whats-the-range-of-the-input-value-desired-to-use-pretrained-resnet152-and-vgg19/1683/2
        output = (x + 1) / 2  # [-1, 1] -> [0, 1]
        if new_torchvision_version:
            output = self.normalize(output)

        for layer in self.model.features[:self.offset]:  # get output from conv4_4
            output = layer(output)
        return output


# VGG_layer = torchvision.models.vgg19(pretrained=True)
# https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/7
VGG_layer = CustomVGG19(pretrained=False).to(device)
jpeg = DiffJPEG(height=data.input_resolution, width=data.input_resolution, differentiable=True, quality=data.JPEG_quality).to(device)
jpeg_50 = DiffJPEG(height=data.input_resolution, width=data.input_resolution, differentiable=True, quality=50).to(device)


def grayscale(x, yuv=False):
    if yuv:
        # Get Y channel from YIQ/YUV format
        return torch.sum(x * grayscale_convert, dim=1)
    else:
        return grayscale_transform(x)


def invertibility_criterion(generated, target):
    return l2Norm(generated, target)


def lightness_criterion(generated, target, threshold=loss_weights.threshold):
    # https://github.com/smartcameras/EdgeFool/blob/master/Train/rgb_lab_formulation_pytorch.py
    # https://discuss.pytorch.org/t/is-there-rgb2lab-function-for-pytorch-data-varaible-or-tensor-or-torch-read-from-an-image/15594
    # https://web.archive.org/web/20120502065620/http://cookbooks.adobe.com/post_Useful_color_equations__RGB_to_LAB_converter-14227.html

    difference = torch.abs(generated - target)
    difference_processed = difference.detach()

    index = (difference_processed <= threshold).float()

    return l1Loss(difference, difference_processed * index)


# def contrast_criterion(generated: torch.Tensor, target: torch.Tensor):
def contrast_criterion(generated, target):
    # resize to 224
    generated = nn.functional.interpolate(generated, size=(224, 224))
    generated = generated.expand(-1, 3, -1, -1)
    target = nn.functional.interpolate(target, size=(224, 224))

    loss = l2Norm(VGG_layer(generated), VGG_layer(target))
    return loss


def get_total_variation(image):
    # https://www.tensorflow.org/api_docs/python/tf/image/total_variation
    # https://github.com/tensorflow/tensorflow/blob/v2.8.0/tensorflow/python/ops/image_ops_impl.py#L3220-L3289
    # https://discuss.pytorch.org/t/implement-total-variation-loss-in-pytorch/55574
    # https://blog.csdn.net/qq_34622844/article/details/88846411
    # https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html
    # https://stats.stackexchange.com/questions/88348/is-variation-the-same-as-variance
    # https://en.wikipedia.org/wiki/Total_variation_denoising

    batch_size, num_of_channels, width, height = image.size()
    variation_x = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).sum()
    variation_y = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).sum()

    return (variation_x + variation_y) / (batch_size * num_of_channels * width * height)


def structure_criterion(generated, target, mean=False):
    variation_generated = get_total_variation(generated)
    variation_target = get_total_variation(target)

    loss = l1Loss(variation_generated, variation_target)

    # variance_generated = torch.var(generated)
    # variance_target = torch.var(target)
    # if mean:
    #     variance_generated = variance_generated / 256 ** 2
    #     variance_target = variance_target / 256 ** 2
    # loss = l1Loss(variance_generated, variance_target)

    return loss


def quantization_criterion(generated, floor_image=False):
    # to_be_implemented
    if floor_image:
        # https://discuss.pytorch.org/t/torch-round-gradient/28628
        # https://tissue333.gitbook.io/cornell/findings/pytorch_backward
        # target = torch.round((generated + 1) * 127.5)
        # target = target / 127.5 - 1
        loss = l1Loss(generated, data.quantize(generated.detach(), use_round=False))
    else:
        batch_size, _, _, _ = generated.size()

        # Potentially large memory usage
        combined = torch.cat([generated for t in range(256)], dim=0)
        ones = torch.ones((batch_size, 1, 1, 1), device=device)
        combined_ones = torch.cat([ones * t for t in range(256)], dim=0)
        combined_ones = (combined_ones / 127.5) - 1
        min_diff = torch.min(torch.abs(combined - combined_ones), dim=0).values
        # min_diff = torch.min(torch.min(torch.abs(combined - combined_ones), dim=3).values, dim=2).values
        loss = torch.mean(min_diff)

    return loss


def apply_dct_artifact(image, grayscale_input=False, quantize=True, quality_50=False):
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
    if grayscale_input:
        image = image.expand(-1, 3, -1, -1)
    if not quality_50:
        image = jpeg(image)
    else:
        image = jpeg_50(image)
    if grayscale_input:
        image = torch.unsqueeze(torch.mean(image, dim=1), dim=1)
    image = image * 2 - 1
    if quantize:
        image = data.quantize(image, use_round=False)
    return image


def apply_downsample_artifact(image, scale, smooth_gradient=True):
    if scale <= 0.0:
        scale = 1.0
    if smooth_gradient:
        forward_value = nn.functional.interpolate(image, scale_factor=1.0/scale, mode='nearest')
        forward_value = nn.functional.interpolate(forward_value, scale_factor=scale, mode='nearest')
        output = image.clone()
        output.data = forward_value.data
    else:
        intermediate = nn.functional.interpolate(image, scale_factor=1.0/scale, mode='nearest')
        output = nn.functional.interpolate(intermediate, scale_factor=scale, mode='nearest')
    return output


def get_MAE_difference(image, target):
    # https://hackernoon.com/my-notes-on-mae-vs-mse-error-metrics
    # https://www.tutorialspoint.com/how-to-measure-the-mean-absolute-error-mae-in-pytorch
    image = (image.detach() + 1) * 127.5
    target = (target.detach() + 1) * 127.5
    return l1Loss(image, target)


def get_SSIM_difference(image, target):
    # https://ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python
    # https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html
    # https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
    # https://blog.csdn.net/Mao_Jonah/article/details/117228926
    # https://github.com/scikit-image/scikit-image/issues/4636
    # https://github.com/Po-Hsun-Su/pytorch-ssim
    # https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html
    # https://www.tensorflow.org/api_docs/python/tf/image/ssim
    return ssim(image.detach(), target.detach())

