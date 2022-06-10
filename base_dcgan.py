# DCGAN code
# Lightly edited from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


def run_dcgan(cmd=False):
    if cmd:
        parser = gen_args()
        opts = parser.parse_args()

        dataset = opts.dataset
        dataroot = opts.dataroot
        workers = opts.workers
        batch_size = opts.batchSize
        image_size = opts.imageSize
        nz = int(opts.nz)
        ngf = int(opts.ngf)
        ndf = int(opts.ndf)
        niter = opts.niter
        lr = opts.lr
        beta1 = opts.beta1
        cuda = opts.cuda
        dry_run = opts.dry_run
        ngpu = int(opts.ngpu)
        load_net_g = opts.netG
        load_net_d = opts.netD5
        outf = opts.outf
        if not opts.manualSeed:
            manualSeed = 483
        else:
            manualSeed = opts.manualSeed
    else:
        dataset = 'folder'
        dataroot = os.path.join('example', 'inputs', 'warpedimagesubset')
        workers = 2
        batch_size = 8
        image_size = 64
        nz = 100
        ngf = 64
        ndf = 64
        niter = 25
        lr = .0002
        beta1 = .5
        cuda = torch.cuda.is_available()
        print(cuda)
        dry_run = True
        ngpu = 1
        outf = os.path.join('example', 'outputs', 'dcgan_results')
        load_net_g = os.path.join(outf, "netG_epoch_10.pth")
        load_net_d = os.path.join(outf, "netD_epoch_10.pth")
        manualSeed = 483

    try:
        os.makedirs(outf)
    except OSError:
        pass

    print(f"Random seed: {manualSeed}")
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True

    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if dataroot is None and str(dataset).lower() != 'fake':
        raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % dataset)

    if dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        nc = 3

    elif dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, image_size, image_size),
                                transform=transforms.ToTensor())
        nc = 3

    else:
        raise ValueError

    assert dataset

    full_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=int(workers))

    device = torch.device("cuda:0" if cuda else "cpu")

    net_g = Generator(ngpu, nz, ngf, nc).to(device)
    net_g.apply(weights_init)
    if load_net_g != '':
        net_g.load_state_dict(torch.load(load_net_g))
    print(net_g)

    net_d = Discriminator(ngpu, nc, ndf).to(device)
    net_d.apply(weights_init)
    if load_net_d != '':
        net_d.load_state_dict(torch.load(load_net_d))
    print(net_d)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))

    if dry_run:
        niter = 1

    for epoch in range(niter):
        for i, full_data in enumerate(full_dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            net_d.zero_grad()
            real_cpu = full_data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                               dtype=real_cpu.dtype, device=device)

            output = net_d(real_cpu)
            err_d_real = criterion(output, label)
            err_d_real.backward()
            d_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = net_g(noise)
            label.fill_(fake_label)
            output = net_d(fake.detach())
            err_d_fake = criterion(output, label)
            err_d_fake.backward()
            d_g_z1 = output.mean().item()
            err_d = err_d_real + err_d_fake
            optimizer_d.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            net_g.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = net_d(fake)
            err_g = criterion(output, label)
            err_g.backward()
            d_g_z2 = output.mean().item()
            optimizer_g.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, niter, i, len(full_dataloader),
                     err_d.item(), err_g.item(), d_x, d_g_z1, d_g_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % outf,
                                  normalize=True)
                fake = net_g(fixed_noise)
                vutils.save_image(fake.detach(),
                                  '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                                  normalize=True)

            if dry_run:
                break
        # do checkpointing
        torch.save(net_g.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
        torch.save(net_d.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))


def weights_init(m):
    """
    custom weights initialization called on netG and netD

    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, inp):
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp):
        if inp.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)

        return output.view(-1, 1).squeeze(1)


def gen_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', required=False, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

    return parser


if __name__ == "__main__":
    run_dcgan()
