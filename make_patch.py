import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from pretrained_models_pytorch import pretrainedmodels

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

parser.add_argument('--target', type=int, default=859, help='The target class: 859 == toaster')
parser.add_argument('--conf_target', type=float, default=0.9, help='Stop attack on image when target classifier reaches this value for target class')

parser.add_argument('--max_count', type=int, default=1000, help='max number of iterations to find adversarial example')
parser.add_argument('--patch_type', type=str, default='circle', help='patch type: circle or square')
parser.add_argument('--patch_size', type=float, default=0.05, help='patch size. E.g. 0.05 ~= 5% of image ')

parser.add_argument('--train_size', type=int, default=2000, help='Number of training images')
parser.add_argument('--test_size', type=int, default=2000, help='Number of test images')

parser.add_argument('--image_size', type=int, default=299, help='the height / width of the input image to network')

parser.add_argument('--plot_all', type=int, default=1, help='1 == plot all successful adversarial images')

parser.add_argument('--netClassifier', default='inceptionv3', help="The target classifier")

parser.add_argument('--outf', default='./logs', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1338, help='manual seed')

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

target = args.target
conf_target = args.conf_target
max_count = args.max_count
transform_patch = args.patch_type
patch_size = args.patch_size
image_size = args.image_size
train_size = args.train_size
test_size = args.test_size
plot_all = args.plot_all

assert train_size + test_size <= 50000, "Traing set size + Test set size > Total dataset size"

print("=> creating model ")
netClassifier = pretrainedmodels.__dict__[args.netClassifier](num_classes=1000, pretrained='imagenet')
if args.cuda:
    netClassifier.cuda()


print('==> Preparing data..')
normalize = transforms.Normalize(mean=netClassifier.mean,
                                 std=netClassifier.std)
idx = np.arange(NUM_OF_CLASSES)
np.random.shuffle(idx)
training_idx = idx[:train_size]
test_idx = idx[train_size:(train_size + test_size)]


train_loader = torch.utils.data.DataLoader(
    dset.ImageFolder(DATA_FOLDER, transforms.Compose([
        transforms.Resize(round(max(netClassifier.input_size)*1.050)),
        transforms.CenterCrop(max(netClassifier.input_size)),
        transforms.ToTensor(),
        ToSpaceBGR(netClassifier.input_space=='BGR'),
        ToRange255(max(netClassifier.input_range)==255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=SubsetRandomSampler(training_idx),
    num_workers=args.workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    dset.ImageFolder(DATA_FOLDER, transforms.Compose([
        transforms.Resize(round(max(netClassifier.input_size)*1.050)),
        transforms.CenterCrop(max(netClassifier.input_size)),
        transforms.ToTensor(),
        ToSpaceBGR(netClassifier.input_space=='BGR'),
        ToRange255(max(netClassifier.input_range)==255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=SubsetRandomSampler(test_idx),
    num_workers=args.workers, pin_memory=True)

min_in, max_in = netClassifier.input_range[0], netClassifier.input_range[1]
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
mean, std = np.array(netClassifier.mean), np.array(netClassifier.std)
min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)


def train(epoch, patch, patch_shape):
    netClassifier.eval()
    success = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)

        prediction = netClassifier(data)
 
        is_model_misclassified = prediction.data.max(1)[1][0] != labels.data[0]
        if is_model_misclassified:
            continue
     
        total += 1

        mask, patch, patch_shape = patch_transformation(data, patch, patch_shape)

        adv_x, mask, patch = attack(data, patch, mask)

        adv_label = netClassifier(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]
        
        if adv_label == target:
            success += 1
      
            if plot_all == 1: 
                # plot source image
                vutils.save_image(data.data, "./%s/%d_%d_%d_original.png" % (args.outf, epoch, batch_idx, ori_label), normalize=True)
                
                # plot adversarial image
                vutils.save_image(adv_x.data, "./%s/%d_%d__%d_adversarial.png" % (args.outf, epoch, batch_idx, adv_label,), normalize=True)
 
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
 
        patch = new_patch

        # log to file  
        progress_bar(batch_idx, len(train_loader), "Train Patch Success: {:.3f}".format(success/total))

    return patch


def patch_transformation(data, patch, patch_shape):
    # transform path
    data_shape = data.data.cpu().numpy().shape
    if transform_patch == 'circle':
        patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
    elif transform_patch == 'square':
        patch, mask = square_transform(patch, data_shape, patch_shape, image_size)
    patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
    if args.cuda:
        patch, mask = patch.cuda(), mask.cuda()
    patch, mask = Variable(patch), Variable(mask)
    return mask, patch, patch_shape


def test(epoch, patch, patch_shape):
    netClassifier.eval()
    success = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)

        prediction = netClassifier(data)

        # only computer adversarial examples on examples that are originally classified correctly        
        if prediction.data.max(1)[1][0] != labels.data[0]:
            continue
      
        total += 1

        mask, patch, patch_shape = patch_transformation(data, patch, patch_shape)

        adv_x = torch.mul((1-mask),data) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)
        
        adv_label = netClassifier(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]
        
        if adv_label == target:
            success += 1
       
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
 
        patch = new_patch

        # log to file  
        progress_bar(batch_idx, len(test_loader), "Test Success: {:.3f}".format(success/total))

def attack(x, patch, mask):
    netClassifier.eval()

    x_out = F.softmax(netClassifier(x))
    target_prob = x_out.data[0][target]

    adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
    
    count = 0 
   
    while conf_target > target_prob:
        count += 1
        adv_x = Variable(adv_x.data, requires_grad=True)
        adv_out = F.log_softmax(netClassifier(adv_x))
       
        adv_out_probs, adv_out_labels = adv_out.max(1)
        
        Loss = -adv_out[0][target]
        Loss.backward()
     
        adv_grad = adv_x.grad.clone()
        
        adv_x.grad.data.zero_()
       
        patch -= adv_grad 
        
        adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)
 
        out = F.softmax(netClassifier(adv_x))
        target_prob = out.data[0][target]
        #y_argmax_prob = out.data.max(1)[0][0]
        
        #print(count, conf_target, target_prob, y_argmax_prob)  

        if count >= args.max_count:
            break


    return adv_x, mask, patch 


if __name__ == '__main__':
    if transform_patch == 'circle':
        patch, patch_shape = init_patch_circle(image_size, patch_size) 
    elif transform_patch == 'square':
        patch, patch_shape = init_patch_square(image_size, patch_size) 
    else:
        sys.exit("Please choose a square or circle patch")
    
    for epoch in range(1, args.epochs + 1):
        patch = train(epoch, patch, patch_shape)
        test(epoch, patch, patch_shape)
