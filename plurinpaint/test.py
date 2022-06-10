import matplotlib.pyplot as plt
import numpy as np

from options import test_options
from dataloader import data_loader
from model import create_model
from util import visualizer
from itertools import islice

if __name__=='__main__':
    # get testing options
    opt = test_options.TestOptions().parse()
    opt.name = 'lower_half_predictor'
    opt.img_file = '../dataset/inputs/warp/'
    opt.mask_file = '../masks/'
    opt.mask_type = [3]
    opt.no_shuffle = True
    opt.no_flip = True
    opt.no_rotation = True
    opt.no_augment = True
    opt.how_many = 200*.8
    opt.nsampling = 2
    opt.save_number = 2
    opt.results_dir = "./results/test/"
    n_samples = int(200 / 8)
    cutoff = int(200 * .8 / 8)
    # creat a dataset
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    model.eval()
    # create a visualizer
    visualizer = visualizer.Visualizer(opt)

    imgs = np.array([])
    sliced = islice(dataset, cutoff, n_samples)
    for data in islice(dataset, cutoff, n_samples):
        model.set_input(data)
        m_p = model.test()
        try:
            imgs = np.append(imgs, m_p, axis=0)
        except ValueError:
            imgs = m_p
    for i in range(len(imgs)):
        ax = plt.subplot(len(imgs), 1, i+1)
        ax.imshow(imgs[i])
    plt.show()
