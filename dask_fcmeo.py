"""
Runs the set of estimators over data
Some changes may need to be made for the concise model to be run on PIC
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import dask
import dask_image.imread
import cv2
import time
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import plurinpaint
from plurinpaint.options import test_options
from plurinpaint.dataloader import data_loader
from plurinpaint.model import create_model
from itertools import islice
from sklearn import metrics

NUM_SAMPLES = 100
IMG_SHAPE = (128, 128)
FACES_TO_PLOT = 10
DS_LOCATION = os.path.join('example', 'inputs', 'warpedimagesubset')
ESTIMATORS = {
    "Linear Regression": LinearRegression(),
    "SVM": SVR(),
    "1K-nn": KNeighborsRegressor(n_neighbors=1, weights='distance'),
    "5K-nn": KNeighborsRegressor(n_neighbors=5, weights='distance'),
    "15K-nn": KNeighborsRegressor(n_neighbors=15, weights='distance'),
    "Decision Tree": DecisionTreeRegressor(max_features='log2'),
    "Random Forest": RandomForestRegressor(n_estimators=10, max_features='log2'),
    "MLP Regressor": MLPRegressor(max_iter=5000)
}
ESTIMATOR_ORDER = [
    'Linear Regression',
    "SVM",
    '1K-nn',
    '5K-nn',
    '15K-nn',
    'Decision Tree',
    'Random Forest',
    'MLP Regressor',
#    'NN1',
#    'NN2'
]

def main():
    mses = []
    maes = []
    run_models(100, False, True)
    print()
    print()


def run_models(num_samples, run_neural_net, plot):
    print("Start")
    start_time = time.perf_counter()
    X_train, y_train, X_test, y_test = get_data('images', num_samples, kind='gs')
    read_time = time.perf_counter()
    print(f'Read finish: {read_time - start_time:.1f}s')
    y_test_predict = dict()
    last_time = time.perf_counter()
    for name, estimator in ESTIMATORS.items():
        print(f'Estimator {name} start')
        try:
            estimator.fit(X_train, y_train)
        except ValueError:
            estimator = MultiOutputRegressor(estimator)
            estimator.fit(X_train.astype('int'), y_train.astype('int'))
        this_time = time.perf_counter()
        print(f'Estimator {name} fit finish: {this_time - last_time:.1f}s')
        last_time = this_time
        y_test_predict[name] = estimator.predict(X_test)
        this_time = time.perf_counter()
        print(f'Estimator {name} predict finish: {this_time - last_time:.1f}s')
        last_time = this_time
    if run_neural_net:
        print(f'Estimator Neural Net start')
        y_test_predict['NN1'], y_test_predict['NN2'] = run_pic_model(num_samples)
        this_time = time.perf_counter()
        print(f'Estimator Neural Net predict finish: {this_time - last_time:.1f}s')
    print(f'All estimator work finished in {last_time - read_time:.1f}s')
    if plot:
        plot_data(X_test, y_test, y_test_predict, num_samples)


def run_pic_model(num_samples, only_faces=False):
    sys.stdout = open(os.devnull, 'w')
    opt = test_options.TestOptions().parse()
    opt.name = 'lower_half_predictor'
    opt.img_file = 'example/inputs/warpedimagesubset/images/'
    opt.mask_file = 'masks/'
    opt.mask_type = [3]
    opt.no_shuffle = True
    opt.no_flip = True
    opt.no_rotation = True
    opt.no_augment = True
    opt.nsampling = 2
    opt.save_number = 2
    opt.results_dir = "./results/output"
    opt.checkpoints_dir = os.path.join('plurinpaint', opt.checkpoints_dir)
    cutoff = int(num_samples * .8)

    # creat a dataset
    dataset = data_loader.dataloader(opt)
    # create a model
    model = create_model(opt)
    model.eval()

    out_data1 = np.array([])
    out_data2 = np.array([])
    if only_faces:
        to_run = islice(dataset, int(cutoff / 8), int(((cutoff + only_faces) / 8) + (only_faces % 8 != 0)))
    else:
        to_run = islice(dataset, int(cutoff / 8), int(num_samples / 8))

    for i, data in enumerate(to_run):
        model.set_input(data)
        out_data = model.test()
        try:
            out_data1 = np.append(out_data1, np.array([i for i in out_data[0:8]]), axis=0)
            out_data2 = np.append(out_data2, np.array([i for i in out_data[8:]]), axis=0)
        except ValueError:
            out_data1 = np.array([i for i in out_data[0:8]])
            out_data2 = np.array([i for i in out_data[8:]])
    sys.stdout = sys.__stdout__
    return out_data1, out_data2


def get_data(folder, num_samples, kind='gs'):
    fn_pattern = os.path.join(DS_LOCATION, folder, f'0*.jpg')
    full_pictures = dask_image.imread.imread(fn_pattern)

    def grayscale(rgb):
        return ((rgb[..., 0] * 0.2126) +
                (rgb[..., 1] * 0.7152) +
                (rgb[..., 2] * 0.0722))

    def rg(rgb):
        return rgb[..., 0] + rgb[..., 1]

    def rb(rgb):
        return rgb[..., 0] + rgb[..., 2]

    if kind == 'gs':
        data = grayscale(full_pictures)
    elif kind == 'rb':
        data = rb(full_pictures)
    elif kind == 'rg':
        data = rg(full_pictures)
    else:
        raise RuntimeError(f'get_data expects gs, rb, or rg. {kind} received instead')
    half = int(IMG_SHAPE[1] / 2)
    cut_off = int(num_samples * .8)
    X_train = data[:cut_off, :half, :].compute()
    y_train = data[:cut_off, half:, :].compute()
    X_test = data[cut_off:num_samples, :half, :].compute()
    y_test = data[cut_off:num_samples, half:, :].compute()
    X_train = [i[:].flatten() for i in X_train]
    y_train = [i[:].flatten() for i in y_train]
    X_test = [i[:].flatten() for i in X_test]
    y_test = [i[:].flatten() for i in y_test]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def plot_data(x_test, y_test, y_test_predict, num_samples):
    n_cols = 1 + len(ESTIMATOR_ORDER)
    fig = plt.figure(figsize=(2.0 * n_cols, 2.26 * FACES_TO_PLOT))
    plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(FACES_TO_PLOT):
        try:
            true_face = np.hstack((x_test[i], y_test[i]))
        except ValueError:
            gray_scaled = np.array([np.array([[[rgb[0] * .2126 + .7152 * rgb[1] + rgb[2] * .0722]
                                               for rgb in x] for x in y]) for y in y_test[i]]).flatten()
            true_face = np.hstack((x_test[i], gray_scaled))

        if i:
            sub = plt.subplot(FACES_TO_PLOT, n_cols, i * n_cols + 1)
        else:
            sub = plt.subplot(FACES_TO_PLOT, n_cols, i * n_cols + 1, title="Original")
        sub.set_yticklabels([])
        sub.set_xticklabels([])
        sub.set_yticks([])
        sub.set_xticks([])
        sub.imshow(
            true_face.reshape(IMG_SHAPE), cmap=plt.cm.gray, interpolation="nearest"
        )
        sub.set_ylabel(f'{int(num_samples * .8) + i:>06}.jpg')

        for j, est in enumerate(ESTIMATOR_ORDER):
            try:
                completed_face = np.hstack((x_test[i], y_test_predict[est][i]))
            except ValueError:
                completed_face = y_test_predict[est][i]

            if i:
                sub = plt.subplot(FACES_TO_PLOT, n_cols, i * n_cols + 2 + j)

            else:
                sub = plt.subplot(FACES_TO_PLOT, n_cols, i * n_cols + 2 + j, title=est)

            sub.set_yticklabels([])
            sub.set_xticklabels([])
            sub.set_yticks([])
            sub.set_xticks([])
            try:
                sub.imshow(
                    completed_face.reshape(IMG_SHAPE),
                    cmap=plt.cm.gray,
                    interpolation="nearest",
                )
            except ValueError:
                img_to_plot = cv2.resize(completed_face, (128, 128))
                sub.imshow(
                    img_to_plot,
                    cmap=plt.cm.gray,
                    interpolation="nearest",
                )

    plt.show()


# def get_error(y_test, y_test_prediction):

def run_models_on_1k():
    print("Start")
    start_time = time.perf_counter()
    X_train, y_train, X_test, y_test = get_big_data('warp', kind='gs')
    read_time = time.perf_counter()
    print(f'Read finish: {read_time - start_time:.1f}s')
    y_test_predict = dict()
    last_time = time.perf_counter()
    for name, estimator in ESTIMATORS.items():
        print(f'Estimator {name} start')
        try:
            estimator.fit(X_train, y_train)
        except ValueError:
            estimator = MultiOutputRegressor(estimator)
            estimator.fit(X_train.astype('int'), y_train.astype('int'))
        this_time = time.perf_counter()
        print(f'Estimator {name} fit finish: {this_time - last_time:.1f}s')
        last_time = this_time
        for i in range(10):
            pred = estimator.predict(X_test[1000*i:1000+1000*i])
            try:
                y_test_predict[name].append(estimator.predict(pred))
            except KeyError:
                y_test_predict[name] = [pred]
        this_time = time.perf_counter()
        print(f'Estimator {name} predict finish: {this_time - last_time:.1f}s')
        last_time = this_time
    print(f'All estimator work finished in {last_time - read_time:.1f}s')
#    y_test_predict['NN1'], y_test_predict['NN2'] = run_pic_model()
#    y_test_predict['NN1'] = run_pic_model()
    for name, predictions in y_test_predict.items():
#        #        try:
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                img = np.hstack((X_test[i*1000 + j], predictions[i][j])).reshape(128, 128)
                cv2.imwrite(f"results\\output\\{name}{i*1000 + j}.jpg", img)


def get_big_data(folder, kind='gs'):
    fn_pattern = os.path.join(DS_LOCATION, folder, f'0*.jpg')
    full_pictures = dask_image.imread.imread(fn_pattern)

    def grayscale(rgb):
        return ((rgb[..., 0] * 0.2126) +
                (rgb[..., 1] * 0.7152) +
                (rgb[..., 2] * 0.0722))

    def rg(rgb):
        return rgb[..., 0] + rgb[..., 1]

    def rb(rgb):
        return rgb[..., 0] + rgb[..., 2]

    if kind == 'gs':
        data = grayscale(full_pictures)
    elif kind == 'rb':
        data = rb(full_pictures)
    elif kind == 'rg':
        data = rg(full_pictures)
    else:
        raise RuntimeError(f'get_data expects gs, rb, or rg. {kind} received instead')
    half = int(IMG_SHAPE[1] / 2)
    X_train = data[:20000, :half, :].compute()
    y_train = data[:20000, half:, :].compute()
    X_test = data[20000:30000, :half, :].compute()
    y_test = data[20000:30000, half:, :].compute()
    X_test = [i[:].flatten() for i in X_test]
    y_test = [i[:].flatten() for i in y_test]
    X_train = [i[:].flatten() for i in X_train]
    y_train = [i[:].flatten() for i in y_train]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


if __name__ == "__main__":
    main()
