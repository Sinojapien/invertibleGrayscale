# import os.path
# import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

from os.path import exists

import architecture as arch
from architecture import data  # import data

import sys
import datetime

# Initial Variables (GAN is very sensitive to these)
current_epochs = 0
current_iter = 0
num_epochs = 0
max_epochs = 120
learning_rate = 0.0002
weight_decay = 0
version = '0'
baseline_flag = ''
use_buffer = True
load_model = True
save_model = True and num_epochs > 5
save_plot = False
use_tensorboard = True
use_lr_scheduler = True
test_model = True
train_stage_two = False

stage_three = True
print(f'Stage three: {stage_three}')
floor_encode_output = True
print(f'Quantize encode output (floor): {floor_encode_output}')
train_compression_artifact = False
print(f'Train jpeg compression artifact: {train_compression_artifact}')
jpeg_probability = 0.0
print(f'Jpeg compression ratio: {jpeg_probability}')
train_downsample_artifact = False
downsample_scale = 4.0/3.0
print(f'Train downsample artifact: {train_downsample_artifact}, {downsample_scale}')
test_jpeg_50 = False
print(f'Test JPEG compression 50 quality: {test_jpeg_50}')
new_test_set = False
print(f'New test set: {new_test_set}')

# Handling input arguments
print(f'Arguments: {sys.argv}')
if len(sys.argv) > 1: data.update_path(sys.argv[1])  # '/data/d0/y19/chyeung9/fyp1/grayscale/'
if len(sys.argv) > 2: data.input_dir = sys.argv[2]  # '/research/dept8/fyp21/ttw2104/'
if len(sys.argv) > 3: num_epochs = int(sys.argv[3])
if len(sys.argv) > 4: version = sys.argv[4]

# Directory
vgg19_model_dir = data.input_dir + 'models/' + 'vgg19/vgg19-dcbb9e9d.pth'
# input_model_dir = data.input_dir + 'models/' + 'InvertibleGrayscale/jpeg_min_w/210/' + 'checkpoint' + version + '.pth'
# input_model_dir = data.input_dir + 'models/' + 'InvertibleGrayscale/new/120_220401/' + 'checkpoint' + version + '.pth'
# input_model_dir = data.input_dir + 'models/' + 'InvertibleGrayscale/jpeg_fix/retrain/150/' + 'checkpoint' + version + '.pth'
# input_model_dir = data.input_dir + 'models/' + 'InvertibleGrayscale/90/' + 'checkpoint' + version + '.pth'
input_model_dir = data.input_dir + 'models/' + 'InvertibleGrayscale/new/120_220401/' + 'checkpoint' + version + '.pth'
print(f'Input Model Directory: {input_model_dir}')
model_dir = data.get_model_directory() + 'checkpoint' + version + '.pth'
print(f'Model Directory: {model_dir}')

# Data section
train_dataloader, test_dataloader = data.get_dataloader(data.input_dir + 'data/voc2012/')
# train_dataloader, _ = data.get_dataloader(data.input_dir + 'data/voc2012/')
n_total_steps = len(train_dataloader)
print('Number of iterations: ' + str(n_total_steps))
# Device Section
device = arch.get_device()
writer = SummaryWriter(
    log_dir=data.get_tensorboard_directory() + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_v-" + version)
# Model Section
if exists(vgg19_model_dir):
    arch.VGG_layer.load_model(vgg19_model_dir)
    print(f'VGG19 model loaded successfully.')
else:
    arch.VGG_layer = arch.CustomVGG19(pretrained=True).to(device)
    print(f'Error: VGG19 model not found.')

encoder = arch.encodeNetwork()
decoder = arch.decodeNetwork()
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.99),
                                     weight_decay=weight_decay)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.99),
                                     weight_decay=weight_decay)

encoder = nn.DataParallel(encoder).to(device)
decoder = nn.DataParallel(decoder).to(device)


def lr_lambda(epoch):
    if max_epochs == 0 or num_epochs == 0:
        return 1.0
    if not stage_three:
        output = 1 - 0.99 * current_epochs / max_epochs
    else:
        output = 1 - 0.99 * epoch / num_epochs
    if output < 0.01:
        output = 0.01
    elif output > 1:
        output = 1.0
    return output
schedulers = [
    lr_scheduler.LambdaLR(encoder_optimizer, lr_lambda=lr_lambda),
    lr_scheduler.LambdaLR(decoder_optimizer, lr_lambda=lr_lambda),
]

# if load_model and exists(model_dir):
#     checkpoint = torch.load(model_dir)
if load_model and exists(input_model_dir):
    checkpoint = torch.load(input_model_dir)
    # checkpoint = torch.load(input_model_dir, map_location = lambda storage, loc: storage)

    current_epochs = checkpoint['epoch']
    current_iter = checkpoint['iteration']

    encoder.load_state_dict(checkpoint['encoder'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
    decoder.load_state_dict(checkpoint['decoder'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

    # if 'scheduler' in checkpoint:
    #     for i in range(len(checkpoint['scheduler'])):
    #         schedulers[i].load_state_dict(checkpoint['scheduler'][i])

    # save memory
    del checkpoint  # checkpoint = {}
    data.clear_cache()
    train_stage_two = current_epochs >= 90  # and not (floor_encode_output or train_compression_artifact)
    print("Loaded model successfully.")

encoder.train()  # .eval()
decoder.train()

# Buffer Section
loss_buffer = arch.LossBuffer()

print()
print('Current Epoch = ' + str(current_epochs))
print(f'Current Iterations = {current_iter}')
print(f'Epochs = {num_epochs}')
print(f'lr = {learning_rate}')
print(f'Weight Decay: {weight_decay}')
print(f'Version = \'{version}\'')
print(f'Baseline Flag = \'{baseline_flag}\'')
print(f'Use Buffer = {use_buffer}')
print(f'Load Model = {load_model}')
print(f'Save Model = {save_model}')
print(f'Use lr_scheduler: {use_lr_scheduler}')
print(f'Test Model = {test_model}')
print(f'Train Stage two: {train_stage_two}')
print(f'Current Time: {datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

# Training Section
for epoch in range(num_epochs):
    train_iter = iter(train_dataloader)

    if train_stage_two:
        if not arch.loss_weights.staged:
            arch.loss_weights.stage()

    for i in range(n_total_steps):
        loss_buffer.clear()

        saveCondition = ((i == n_total_steps - 1) or i == 0) and (epoch % 5 == 0)
        printCondition = (i + 1) % int(n_total_steps) == 0 or i == 0
        addScalarCondition = (i + 1) % int(n_total_steps / 5) == 0 or i == 0

        if printCondition:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}]')

        train_image, _ = next(train_iter)
        train_image = train_image.to(device)

        train_image_gray = arch.grayscale(train_image, yuv=not arch.new_torchvision_version)
        encoded_image = encoder(train_image.detach())

        if train_compression_artifact:
            # compress = jpeg_probability <= 0 or np.random.rand() < jpeg_probability
            # if compress:

            train_image_jpeg = arch.apply_dct_artifact(train_image)
            # train_image_gray_jpeg = arch.apply_dct_artifact(train_image_gray)

            encoded_image_jpeg = arch.apply_dct_artifact(encoded_image, grayscale_input=True)

            decoded_image_jpeg = decoder(encoded_image_jpeg)

            invertibility_loss_jpeg = arch.invertibility_criterion(decoded_image_jpeg, train_image_jpeg.detach()) * arch.loss_weights.inv
            loss_buffer.decoder_loss = loss_buffer.decoder_loss + invertibility_loss_jpeg * arch.loss_weights.jpeg

            # lightness_loss_jpeg = arch.lightness_criterion(encoded_image_jpeg, train_image_gray.detach()) * arch.loss_weights.light
            # contrast_loss_jpeg = arch.contrast_criterion(encoded_image_jpeg, train_image.detach()) * arch.loss_weights.con
            # structure_loss_jpeg = arch.structure_criterion(encoded_image_jpeg, train_image_gray.detach()) * arch.loss_weights.struct
            # confirmity_loss_jpeg = lightness_loss_jpeg + contrast_loss_jpeg + structure_loss_jpeg

            # loss_buffer.encoder_loss = loss_buffer.encoder_loss + confirmity_loss_jpeg * arch.loss_weights.conf

            if addScalarCondition and use_tensorboard:
                writer.add_scalar("Train/Invertibility/JPEG", invertibility_loss_jpeg, current_iter)
                # writer.add_scalar("Train/Lightness/JPEG", lightness_loss_jpeg, current_iter)
                # writer.add_scalar("Train/Contrast/JPEG", contrast_loss_jpeg, current_iter)
                # writer.add_scalar("Train/Structure/JPEG", structure_loss_jpeg, current_iter)

            if saveCondition and use_tensorboard:
                # writer.add_images('Train/INPUT/JPEG', train_image_jpeg / 2 + 0.5, current_iter)
                # writer.add_images('Train/OUTPUT/ENCODE/JPEG', encoded_image_jpeg / 2 + 0.5, current_iter)
                writer.add_images('Train/OUTPUT/DECODE/JPEG', decoded_image_jpeg / 2 + 0.5, current_iter)

            # Free memory
            invertibility_loss_jpeg = None
            train_image_jpeg = None
            encoded_image_jpeg = None
            decoded_image_jpeg = None

        if train_downsample_artifact:
            train_image_downsample = arch.apply_downsample_artifact(train_image, scale=downsample_scale)
            encoded_image_downsample = arch.apply_downsample_artifact(encoded_image, scale=downsample_scale)
            decoded_image_downsample = decoder(encoded_image_downsample)

            invertibility_loss_downsample = arch.invertibility_criterion(decoded_image_downsample, train_image_downsample.detach()) * arch.loss_weights.inv
            loss_buffer.decoder_loss = loss_buffer.decoder_loss + invertibility_loss_downsample * arch.loss_weights.down

            if addScalarCondition and use_tensorboard:
                writer.add_scalar("Train/Invertibility/DOWN", invertibility_loss_downsample, current_iter)

            if saveCondition and use_tensorboard:
                # writer.add_images('Train/OUTPUT/ENCODE/DOWN', encoded_image_downsample / 2 + 0.5, current_iter)
                writer.add_images('Train/OUTPUT/DECODE/DOWN', decoded_image_downsample / 2 + 0.5, current_iter)

            # Free memory
            invertibility_loss_downsample = None
            train_image_downsample = None
            encoded_image_downsample = None
            decoded_image_downsample = None

        if floor_encode_output:
            encoded_image = data.quantize(encoded_image, use_round=False)
        decoded_image = decoder(encoded_image)

        invertibility_loss = arch.invertibility_criterion(decoded_image, train_image.detach()) * arch.loss_weights.inv
        loss_buffer.decoder_loss = loss_buffer.decoder_loss + invertibility_loss

        lightness_loss = arch.lightness_criterion(encoded_image, train_image_gray.detach()) * arch.loss_weights.light
        contrast_loss = arch.contrast_criterion(encoded_image, train_image.detach()) * arch.loss_weights.con
        structure_loss = arch.structure_criterion(encoded_image, train_image_gray.detach()) * arch.loss_weights.struct
        confirmity_loss = lightness_loss + contrast_loss + structure_loss
        loss_buffer.encoder_loss = loss_buffer.encoder_loss + confirmity_loss * arch.loss_weights.conf

        if train_stage_two and not floor_encode_output:
            quantization_loss = arch.quantization_criterion(encoded_image, floor_image=False) * arch.loss_weights.quan
            loss_buffer.encoder_loss = loss_buffer.encoder_loss + quantization_loss
            if addScalarCondition and use_tensorboard:
                writer.add_scalar("Train/Quantization", quantization_loss, current_iter)

        if addScalarCondition and use_tensorboard:
            writer.add_scalar("Train/Invertibility", invertibility_loss, current_iter)
            writer.add_scalar("Train/Lightness", lightness_loss, current_iter)
            writer.add_scalar("Train/Contrast", contrast_loss, current_iter)
            writer.add_scalar("Train/Structure", structure_loss, current_iter)

        if printCondition:
            loss_buffer.print_info()

        if saveCondition and use_tensorboard:
            writer.add_images('Train/INPUT', train_image / 2 + 0.5, current_iter)
            # writer.add_images('Train/INPUT/GRAYSCALE', train_image_gray / 2 + 0.5, current_iter)
            writer.add_images('Train/OUTPUT/ENCODE', encoded_image / 2 + 0.5, current_iter)
            writer.add_images('Train/OUTPUT/DECODE', decoded_image / 2 + 0.5, current_iter)

        # Free memory
        train_image = None
        train_image_gray = None
        encoded_image = None
        decoded_image = None
        invertibility_loss = None
        lightness_loss = None
        contrast_loss = None
        structure_loss = None
        quantization_loss = None

        # Backward
        total_loss = loss_buffer.encoder_loss + loss_buffer.decoder_loss
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        total_loss.backward(retain_graph=False)
        encoder_optimizer.step()
        decoder_optimizer.step()

        current_iter += 1

    writer.add_scalar("LR", schedulers[0].get_last_lr()[0], current_epochs)
    for scheduler in schedulers:
        scheduler.step()

    current_epochs += 1
    print(f'Current Time: {datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

print('Finished Training')

if save_plot:
    data.plot_all(str(current_epochs - num_epochs + 1) + '_' + str(current_epochs))

if save_model:
    torch.save({
        'epoch': current_epochs,
        'iteration': current_iter,

        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),

        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict(),

        'scheduler': [
            schedulers[0].state_dict(),
            schedulers[1].state_dict(),
        ]

    }, model_dir)

    print(f'Saved models as \"{model_dir}\"')

if test_model:
    # Free Memory
    # https://pytorch.org/docs/stable/notes/faq.html
    # https://discuss.pytorch.org/t/how-can-i-release-the-unused-gpu-memory/81919/5
    # https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879/3
    del train_dataloader  # train_dataloader = None
    del encoder_optimizer  # encoder_optimizer = None
    del decoder_optimizer  # decoder_optimizer = None
    schedulers = []
    if num_epochs > 0:
        if use_tensorboard:
            writer.flush()
    data.clear_cache()

    MAE_float_buffer = []
    SSIM_float_buffer = []

    if new_test_set:
        test_dataloader = data.get_image_folder(data.input_dir + 'data/InvertibleGrayscale/test/')

    if train_stage_two:
        arch.loss_weights.stage()
    encoder.eval()
    decoder.eval()

    test_iter = iter(test_dataloader)
    total_test_steps = len(test_dataloader)
    print(f'Total testing steps: {total_test_steps}')

    with torch.no_grad():
        for i in range(total_test_steps):

            saveCondition = True
            addScalarCondition = True
            if total_test_steps > 100:
                saveCondition = ((i + 1) % int(total_test_steps / 5)) == 0 or i == 0
                addScalarCondition = ((i + 1) % int(total_test_steps / 50)) == 0 or i == 0

            test_image, _ = next(test_iter)
            test_image = test_image.to(device)
            test_image_gray = arch.grayscale(test_image, yuv=not arch.new_torchvision_version)

            test_encoded_image = encoder(test_image)

            # Train JPEG
            if train_compression_artifact and i < total_test_steps / 2:  # np.random.rand() < 0.5
                test_image_jpeg = arch.apply_dct_artifact(test_image)
                test_encoded_image_jpeg = arch.apply_dct_artifact(test_encoded_image, grayscale_input=True, quality_50=test_jpeg_50)
                test_decoded_image_jpeg = decoder(test_encoded_image_jpeg)

                if saveCondition and use_tensorboard:
                    writer.add_images('Test/INPUT/JPEG', data.quantize(test_image_jpeg, use_round=False) / 2 + 0.5, i)
                    writer.add_images('Test/OUTPUT/ENCODE/JPEG', test_encoded_image_jpeg / 2 + 0.5, i)
                    writer.add_images('Test/OUTPUT/DECODE/JPEG', test_decoded_image_jpeg / 2 + 0.5, i)

                test_image_jpeg = None
                test_encoded_image_jpeg = None
                test_decoded_image_jpeg = None

            # Train Downsample
            if train_downsample_artifact and i < total_test_steps > 2:  # np.random.rand() < 0.5
                test_image_downsample = arch.apply_downsample_artifact(test_image, scale=downsample_scale)
                test_encoded_image_downsample = arch.apply_downsample_artifact(test_encoded_image, scale=downsample_scale)
                test_decoded_image_downsample = decoder(test_encoded_image_downsample)

                if saveCondition and use_tensorboard:
                    writer.add_images('Test/INPUT/DOWN', data.quantize(test_image_downsample, use_round=False) / 2 + 0.5, i)
                    writer.add_images('Test/OUTPUT/ENCODE/DOWN', test_encoded_image_downsample / 2 + 0.5, i)
                    writer.add_images('Test/OUTPUT/DECODE/DOWN', test_decoded_image_downsample / 2 + 0.5, i)

                test_image_downsample = None
                test_encoded_image_downsample = None
                test_decoded_image_downsample = None

            # Quantization
            encoded_image_quant = data.quantize(test_encoded_image, use_round=False)
            test_decoded_image = decoder(encoded_image_quant)

            invertibility_loss = arch.invertibility_criterion(test_decoded_image, test_image) * arch.loss_weights.inv

            lightness_loss = arch.lightness_criterion(encoded_image_quant, test_image_gray)  # * arch.loss_weights.light
            contrast_loss = arch.contrast_criterion(encoded_image_quant, test_image)  # * arch.loss_weights.con
            structure_loss = arch.structure_criterion(encoded_image_quant, test_image_gray)  # * arch.loss_weights.struct
            quantization_loss = arch.quantization_criterion(test_encoded_image, floor_image=True)  # * arch.loss_weights.quan

            if addScalarCondition and use_tensorboard:
                writer.add_scalar("Test/Invertibility", invertibility_loss, i)
                writer.add_scalar("Test/Lightness", lightness_loss, i)
                writer.add_scalar("Test/Contrast", contrast_loss, i)
                writer.add_scalar("Test/Structure", structure_loss, i)
                writer.add_scalar("Test/Quantization", quantization_loss, i)

            if saveCondition and use_tensorboard:
                writer.add_images('Test/INPUT', test_image / 2 + 0.5, i)
                writer.add_images('Test/INPUT/GRAYSCALE', test_image_gray / 2 + 0.5, i)
                writer.add_images('Test/OUTPUT/ENCODE', test_encoded_image / 2 + 0.5, i)
                writer.add_images('Test/OUTPUT/DECODE', test_decoded_image / 2 + 0.5, i)

            MAE_float_buffer.append(arch.get_MAE_difference(test_decoded_image, test_image).item())
            SSIM_float_buffer.append(arch.get_SSIM_difference(test_decoded_image, test_image).item())

    print('Finished Testing')

    print()
    print(f'Length: {len(MAE_float_buffer)}, {len(SSIM_float_buffer)}')
    print(f"MAE: {data.get_mean(MAE_float_buffer)}, {data.get_stddev(MAE_float_buffer)}")
    print(f"SSIM: {data.get_mean(SSIM_float_buffer)}, {data.get_stddev(SSIM_float_buffer)}")
    print()

if use_tensorboard:
    writer.flush()
    writer.close()
    print('Flushed to Tensorboard')

# tensorboard --logdir=/data/d0/y19/chyeung9/fyp1/submit/outputs/tensorboard/ --host=137.189.88.87 --port=6006
# tensorboard --logdir=C:\Users\User\PycharmProjects\pixelization\tensorboard --port=6006
# tensorboard --logdir=C:\Users\Lenovo\PycharmProjects\pixelization\tensorboard --port=6006
# tensorboard --logdir=C:\Users\Sinojapien\PycharmProjects\pixelization\tensorboard --port=6006

# evaluation, difference map, [0, 32], target of jpeg is OG input or jpeged input???
