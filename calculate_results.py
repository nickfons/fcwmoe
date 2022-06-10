"""
Calculates MSE and MAE
Only uses past data for the concise project
"""

import dask
import dask_image.imread
import dask_image
import os
import matplotlib.pyplot as plt
import cv2
import dask.array
import skimage.metrics
import numpy as np

OG_DS_LOCATION = os.path.join('dataset', 'example/inputs')
DS_LOCATION = os.path.join('results', 'output')


def crop_and_grayscale():
    pattern = os.path.join(DS_LOCATION, 'pic', f"*")
    pics = dask_image.imread.imread(pattern)

    def grayscale(rgb):
        return ((rgb[..., 0] * 0.2126) +
                (rgb[..., 1] * 0.7152) +
                (rgb[..., 2] * 0.0722))

    pics = grayscale(pics).compute()

    for i in range(len(pics)):
        to_save = cv2.resize(pics[i], (128, 128))
        cv2.imwrite(os.path.join(DS_LOCATION, 'pic_gsc', f'{i}.jpg'), to_save)

def calculate_errors():
    true_pattern = os.path.join(os.path.join(OG_DS_LOCATION, 'cropped'), f'0*.jpg')
    true_pictures = dask_image.imread.imread(true_pattern)

    def grayscale(rgb):
        return ((rgb[..., 0] * 0.2126) +
                (rgb[..., 1] * 0.7152) +
                (rgb[..., 2] * 0.0722))

    true_pictures = grayscale(true_pictures).astype(int)
    true_pictures = true_pictures[20000:30000, 64:, :]

    folders = ['dectree', 'mlp', 'neigh1', 'neigh5', 'neigh15', 'ranfor', 'pic_gsc']
    maes = {}
    mses = {}
    psnrs = {}
    ssis = {}
    for i in folders:
        if i == 'pic':
            f_pattern = os.path.join(DS_LOCATION, i, f'*.png')
            this_pictures = dask_image.imread.imread(f_pattern)

            this_pictures = grayscale(this_pictures)
        else:
            f_pattern = os.path.join(DS_LOCATION, i, f'*.jpg')
            this_pictures = dask_image.imread.imread(f_pattern)

        this_pictures = this_pictures[:, 64:, :]

        maes[i] = []
        mses[i] = []
        psnrs[i] = []
        ssis[i] = []
        def get_psnrs(true, pred):
            return skimage.metrics.peak_signal_noise_ratio(true[...].astype(int), pred[...].astype(int))

        def get_ssis(true, pred):
            return skimage.metrics.structural_similarity(true[...].astype(int), pred[...].astype(int))

        for j in range(10):
            print(j)
#            mae = metrics.mean_absolute_error(true_pictures[j*1000:(j+1)*1000], this_pictures[j*1000:(j+1)*1000])
#            mse = metrics.mean_squared_error(true_pictures[j*1000:(j+1)*1000], this_pictures[j*1000:(j+1)*1000])
            psnr_arr = get_psnrs(true_pictures[j*1000:(j+1)*1000], this_pictures[j*1000:(j+1)*1000])
            ssi_arr = get_ssis(true_pictures[j*1000:(j+1)*1000], this_pictures[j*1000:(j+1)*1000])
#            maes[i].append(mae)
#            mses[i].append(mse)
            psnrs[i].append(np.mean(psnr_arr))
            ssis[i].append(np.mean(ssi_arr))

    return maes, mses, psnrs, ssis

if __name__ == "__main__":
    #crop_and_grayscale()
#    mae, mses, psnrs, ssis = calculate_errors()
#    print(psnrs)
#    print(ssis)
    pre_calc_mae = {'dectree': [53.73702106813965, 55.002647908886715, 54.554674498681635, 56.583252340063474, 55.26738128679199, 55.84300322270508, 53.76914362265625, 54.552530322265625, 54.538129758886726, 54.70879730808105], 'mlp': [49.68425986130371, 49.479671382861326, 49.46216379267578, 51.27425766159668, 50.004988630590816, 50.604417308642574, 49.301476638867186, 49.46981415424804, 49.12647847041015, 49.23637954572754], 'neigh1': [51.25827987917481, 50.216530478710936, 51.92652861425781, 52.027444899194336, 51.52573917263183, 51.775889474218744, 51.032759095458985, 51.49924414345703, 50.95979728623047, 50.678954024145504], 'neigh5': [45.89992200212402, 44.75533647827148, 45.770080942187505, 47.24433725671386, 45.57417185012207, 46.18594603867187, 45.51745939975586, 45.327885431591795, 45.01461078886719, 44.81121898537597], 'neigh15': [44.65841012224121, 43.67527842504883, 44.5827262796875, 46.05938189763184, 44.37829211179199, 45.053099447949215, 44.1807533024414, 44.004155082861324, 44.212963379248045, 43.81160247785644], 'ranfor': [44.3282982697998, 43.72836283286133, 44.54199187255859, 45.78356360314941, 44.35379077243652, 44.985491407470704, 44.155188613330075, 43.93732500209961, 44.104245887353514, 43.66837519396972], 'pic_gsc': [51.34666614533691, 49.591262914111326, 51.331447895947264, 52.902299745434576, 50.524507914233396, 50.68645274565429, 50.427903893212886, 50.56237266328125, 50.9968932184082, 50.531550235375974]}
    pre_calc_mse = {'dectree': [4670.3779357981475, 4863.825476793044, 4760.16339359658, 5087.126526318023, 4879.857870924448, 4998.723126560906, 4667.869727925425, 4772.142639031101, 4776.370052032033, 4804.054433301395], 'mlp': [3848.0430659185577, 3794.6656069600845, 3800.122779102928, 4045.4538379431206, 3876.980112561362, 3935.269339666863, 3786.626925692173, 3766.341905341404, 3738.8660393497084, 3764.2129842364525], 'neigh1': [4262.214044521243, 4060.4875331770763, 4319.544975979392, 4311.599975391021, 4205.767124188511, 4312.866333358757, 4183.2308756765, 4253.870506164743, 4171.801102160255, 4122.58659456824], 'neigh5': [3293.487103461429, 3124.0695986205337, 3232.392639282908, 3422.147756289556, 3209.015740805845, 3272.8348206920095, 3188.866044242807, 3166.492717346873, 3124.8379156548353, 3114.182704563503], 'neigh15': [3089.9350007639678, 2947.802670075221, 3037.8790610954566, 3222.465487510357, 3020.5510183287943, 3085.4343530476244, 2986.0664791564795, 2957.880014630613, 2982.592981905812, 2947.584446002566], 'ranfor': [3023.126034752103, 2929.0582823213153, 3027.8081208795875, 3176.4908570738335, 3004.093636746128, 3078.6293996008962, 2981.9544453266944, 2932.7378296723123, 2958.176971652687, 2922.9714249686303], 'pic_gsc': [4239.19564876724, 4009.709015647292, 4234.787209605125, 4431.14563755484, 4105.221962822983, 4158.677291376483, 4098.075268255356, 4116.973267151365, 4203.085248074709, 4136.189455870486]}
    folders = ['neigh1', 'neigh5', 'neigh15', 'dectree', 'ranfor', 'mlp', 'pic_gsc']
    nice_names = ['K-NN\n1', 'K-NN\n5', 'K-NN\n15', "Dec.\nTree", 'Ran.\nForest', 'MLP', "PIC"]
    values = [pre_calc_mae[k] for k in folders]

    plt.boxplot(values)
    plt.xticks(range(1, len(nice_names) + 1), nice_names)
    plt.ylabel('Mean Absolute Error')
    plt.savefig('mae.svg', dpi=300)
    plt.show()

    values = [pre_calc_mse[k] for k in folders]
    plt.boxplot(values)
    plt.xticks(range(1, len(nice_names) + 1), nice_names)
    plt.ylabel('Mean Squared Error')
    plt.savefig('mse.svg', dpi=300)
