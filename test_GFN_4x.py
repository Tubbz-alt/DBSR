# Citation:
#     Gated Fusion Network for Joint Image Deblurring and Super-Resolution
#     The British Machine Vision Conference(BMVC2018 oral)
#     Xinyi Zhang, Hang Dong, Zhe Hu, Wei-Sheng Lai, Fei Wang and Ming-Hsuan Yang
# Contact:
#     cvxinyizhang@gmail.com
# Project Website:
#     http://xinyizhang.tech/bmvc2018
#     https://github.com/jacquelinelala/GFN

from __future__ import print_function
import argparse
import os
import time
from math import log10
from os.path import join
from torchvision import transforms
from torchvision import utils as utils
import torch
from torch.utils.data import DataLoader
from datasets.dataset_hf5 import DataValSet
import statistics
import matplotlib.pyplot as plot
import re
from skimage.metrics import structural_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_pkl(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])


def which_trainingstep_epoch(resume):
    trainingstep = "".join(re.findall(r"\d", resume)[0])
    start_epoch = "".join(re.findall(r"\d", resume)[1:])
    return int(trainingstep), int(start_epoch)


def displayFeature(feature):
    feat_permute = feature.permute(1, 0, 2, 3)
    grid = utils.make_grid(feat_permute.cpu(), nrow=16, normalize=True, padding=10)
    grid = grid.numpy().transpose((1, 2, 0))
    display_grid = grid[:, :, 0]
    plot.imshow(display_grid)


def test(test_gen, model, criterion, SR_dir, opt):
    avg_psnr = 0
    avg_ssim = 0
    med_time = []

    with torch.no_grad():
        for iteration, batch in enumerate(test_gen, 1):
            LR_Blur = batch[0].to(device)
            HR = batch[1].to(device)

            test_Tensor = torch.cuda.FloatTensor().resize_(1).zero_() + 1
            if opt.gated == True:
                gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_() + 1
            else:
                gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()

            start_time = time.perf_counter()  # -------------------------begin to deal with an image's time
            _, _, sr = model(LR_Blur, gated_Tensor, test_Tensor)
            # modify
            sr = torch.clamp(sr, min=0, max=1)
            print(f"SR shape is {sr.shape} hr shape is {HR.shape}")
            torch.cuda.synchronize(device)  # wait for CPU & GPU time syn
            evalation_time = time.perf_counter() - start_time  # ---------finish an image
            med_time.append(evalation_time)

            resultSRDeblur = transforms.ToPILImage()(sr.cpu()[0])
            resultSRDeblur.save(join(SR_dir, '{0:04d}_GFN_4x.png'.format(iteration)))
            print("Processing {}".format(iteration))
            mse = criterion(sr, HR)
            psnr = 10 * log10(1 / mse)
            ssim = structural_similarity(sr, HR[0], multichannel=True)
            avg_psnr += psnr
            avg_ssim += ssim

        print("Avg. SR PSNR:{:4f} dB  Avg. SR SSIM:{:4f} dB".format(avg_psnr / iteration, avg_ssim / iteration))
        median_time = statistics.median(med_time)
        print(median_time)


def model_test(model, testloader, SR_dir, opt):
    model = model.to(device)
    criterion = torch.nn.MSELoss(size_average=True)
    criterion = criterion.to(device)
    print(opt)
    test(testloader, model, criterion, SR_dir, opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
    parser.add_argument("--gated", type=bool, default=True, help="Activated gate module")
    parser.add_argument("--isTest", type=bool, default=True, help="Test or not")
    parser.add_argument('--dataset', required=True, help='Path of the validation dataset')
    parser.add_argument("--final_model", default="/model", type=str)
    parser.add_argument("--intermediate_process", default="", type=str, help="Test on intermediate pkl (default: none)")

    test_set = [
        {'gated': False},
        {'gated': False},
        {'gated': True}
    ]
    opt = parser.parse_args()
    root_val_dir = opt.dataset  # #----------Validation path
    SR_dir = join(root_val_dir, 'Results')  # --------------------------SR results save path
    isexists = os.path.exists(SR_dir)
    if not isexists:
        os.makedirs(SR_dir)
    print("The results of testing images stored in {}.".format(SR_dir))

    testloader = DataLoader(DataValSet(root_val_dir), batch_size=1, shuffle=False, pin_memory=False)
    print("===> Loading model and criterion")

    if opt.intermediate_process:
        test_pkl = opt.intermediate_process
        if is_pkl(test_pkl):
            print("Testing model {}----------------------------------".format(opt.intermediate_process))
            train_step, epoch = which_trainingstep_epoch(opt.intermediate_process)
            opt.gated = test_set[train_step - 1]['gated']
            model = torch.load(test_pkl, map_location=lambda storage, loc: storage)
            model_test(model, testloader, SR_dir, opt)
        else:
            print("--opt.intermediate_process /models/1/GFN_epoch_25.pkl)")
    else:
        test_dir = 'models/'
        test_list = [x for x in sorted(os.listdir(test_dir)) if is_pkl(x)]
        print("Testing on the given 3-step trained model which stores in /models, and ends with pkl.")
        for i in range(len(test_list)):
            print("Testing model is {}----------------------------------".format(test_list[i]))
            model = torch.load(join(test_dir, test_list[i]), map_location=lambda storage, loc: storage)
            model_test(model, testloader, SR_dir, opt)
