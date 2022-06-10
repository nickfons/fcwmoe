import os.path

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.nn.functional

DATASET_ROOT = "dataset"
DATASET_INPUTS = 'inputs'
DATASET_OUTPUTS = 'outputs'
FULL_CELEB_NAME = 'full'
LH_CELEB_NAME = 'lower_half'
UH_CELEB_NAME = 'upper_half'
ALL_CELEB_NAME = 'all'
FULL_CELEB_IMAGES = os.path.join(DATASET_ROOT, FULL_CELEB_NAME)
LH_CELEB_IMAGES = os.path.join(DATASET_ROOT, DATASET_OUTPUTS, LH_CELEB_NAME)
UH_CELEB_IMAGES = os.path.join(DATASET_ROOT, DATASET_INPUTS, UH_CELEB_NAME)
ALL_CELEB_IMAGES = os.path.join(DATASET_ROOT, DATASET_INPUTS, ALL_CELEB_NAME)
UH_NDS_NAME = "uh_fn_and_points.csv"
LH_NDS_NAME = "lh_fn_and_points.csv"
ALL_NDS_NAME = "all_fn_and_points.csv"
UH_CSV = os.path.join(DATASET_ROOT, DATASET_OUTPUTS, UH_NDS_NAME)
LH_CSV = os.path.join(DATASET_ROOT, DATASET_INPUTS, LH_NDS_NAME)
ALL_CSV = os.path.join(DATASET_ROOT, DATASET_INPUTS, ALL_NDS_NAME)
MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles
MP_FACE_MESH = mp.solutions.face_mesh


def generate_face_data(image_fn, plot=False, img_write=False, make_dataset=False):
    image = cv2.imread(image_fn)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_x = image.shape[1]
    image_y = image.shape[0]

    try:
        norm_landmarks, mesh_image = generate_mp_landmark(image, plot)
    except TypeError:
        return

    real_landmarks = np.array([[lm[0] * image_x, lm[1], lm[2] * image_y] for lm in norm_landmarks])
    norm_tensor = torch.tensor(norm_landmarks)
    norm_means = norm_tensor.mean(dim=-2, keepdim=True)
    norm_stds = norm_tensor.std(dim=-2, keepdim=True)
    stan_landmarks = (norm_tensor - norm_means) / norm_stds

    stan_landmarks = torch.nn.functional.normalize(stan_landmarks, dim=1)
    stan_landmarks = stan_landmarks.numpy()
#    print(np.max(stan_landmarks[:, 0]), np.max(stan_landmarks[:, 1]), np.max(stan_landmarks[:, 2]))
#    print(np.min(stan_landmarks[:, 0]), np.min(stan_landmarks[:, 1]), np.min(stan_landmarks[:, 2]))
    lr_line, _ = generate_regression(real_landmarks, image_x)
#    stlr_line, stlr_regr = generate_regression(stan_landmarks, image_x)
#    num_over = 0
#    for i in stan_landmarks:
#        if stlr_regr.predict([[i[0]]]) > i[2]:
#            num_over += 1
#    print(num_over)

#    with open(os.path.join(DATASET_ROOT, DATASET_INPUTS, 'normed.csv'), 'w') as nm:
#        nm.write("fname\n")
#    with open(os.path.join(DATASET_ROOT, DATASET_INPUTS, 'normed.csv'), 'a') as nm:
#        nm.write(os.path.split(image_fn)[-1])
#        nm.write(f',{",".join([str(mean) for mean in norm_means[0]])},{",".join([str(std) for std in norm_stds[0]])}')
#        print(f',{",".join([str(mean) for mean in norm_means[0]])},{",".join([str(std) for std in norm_stds[0]])}')
#        all_points = []
#        for i in range(len(stan_landmarks[:, 0])):
#            all_points.append(f'{stan_landmarks[i][0]:.6f},{stan_landmarks[i][1]:.6f},{stan_landmarks[i][2]:.6f}')
#        nm.write(",".join(all_points) + '\n')


    if img_write:
        uh_image = remove_image_lower_half(image, image_x, image_y, lr_line)
        uh_image = cv2.cvtColor(uh_image, cv2.COLOR_RGB2BGR)
        lh_image = remove_image_upper_half(image, image_x, lr_line)
        lh_image = cv2.cvtColor(lh_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(UH_CELEB_IMAGES, image_fn[-10:]), uh_image)
        cv2.imwrite(os.path.join(LH_CELEB_IMAGES, image_fn[-10:]), lh_image)
        cv2.imwrite(os.path.join(ALL_CELEB_IMAGES, image_fn[-10:]), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if plot:
        uh_image = remove_image_lower_half(image, image_x, image_y, lr_line)
        lh_image = remove_image_upper_half(image, image_x, lr_line)
        plot_data(image, mesh_image, uh_image, lh_image, real_landmarks, lr_line, stan_landmarks)

    if make_dataset:
        above_pts = {}
        below_pts = {}
        all_data = []
        with open(UH_CSV, 'a') as uh_nds,\
                open(LH_CSV, 'a') as lh_nds,\
                open(ALL_CSV, 'a') as all_nds:
            uh_nds.write(os.path.split(image_fn)[-1])
            lh_nds.write(os.path.split(image_fn)[-1])
            all_nds.write(os.path.split(image_fn)[-1])
            try:
                lm_gen = lm_analysis(norm_landmarks, real_landmarks, lr_line)
            except:
                print('???', image_fn)
                return
            for [above, idx, lm_data] in lm_gen:
                if above:
                    uh_nds.write(f",{idx},{lm_data[0]:.6f},{lm_data[1]:.6f},{lm_data[2]:.6f}")
                    above_pts[idx] = lm_data
                else:
                    lh_nds.write(f",{idx},{lm_data[0]:.6f},{lm_data[1]:.6f},{lm_data[2]:.6f}")
                    below_pts[idx] = lm_data
                all_nds.write(f",{lm_data[0]:.6f},{lm_data[1]:.6f},{lm_data[2]:.6f}")
                #for i in lm_data:
                all_data.append(lm_data[0]*image_x)
                all_data.append(lm_data[1])
                all_data.append(lm_data[2]*image_y)

            uh_nds.write("\n")
            lh_nds.write("\n")
            all_nds.write("\n")
        return above_pts, below_pts, all_data


    return


def generate_mp_landmark(image_fp, gen_image=False):
    """
    Generates a 3D array of face landmark points by analyzing a given
    image with Google's Mediapipe Library

    :param image_fp: cv2 image variable
    :param gen_image: bool, whether to generate an annotated image
    :return: landmarks: 3D Numpy array of normalized landmarks
             mesh_image: annotated image, None if gen_image is False
    """
    mesh_image = None
    with MP_FACE_MESH.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False,
                               min_detection_confidence=.1) as face_mesh:
        multi_face = face_mesh.process(image_fp)
        try:
            face_landmarks = multi_face.multi_face_landmarks[0]
        except TypeError as e:
            raise e

        if gen_image:
            mesh_image = image_fp.copy()
            MP_DRAWING.draw_landmarks(
                image=mesh_image,
                landmark_list=face_landmarks,
                connections=MP_FACE_MESH.FACEMESH_TESSELATION,
                connection_drawing_spec=MP_DRAWING_STYLES.get_default_face_mesh_tesselation_style())
            MP_DRAWING.draw_landmarks(
                image=mesh_image,
                landmark_list=face_landmarks,
                connections=MP_FACE_MESH.FACEMESH_CONTOURS,
                connection_drawing_spec=MP_DRAWING_STYLES.get_default_face_mesh_contours_style())

    landmarks = np.array([[landmark.x, landmark.z, landmark.y] for landmark in face_landmarks.landmark])

    return landmarks, mesh_image


def generate_regression(landmarks, x_range):
    """
    Generates a regression line of the landmarks on a face

    :param landmarks: 3D NP Array, representing lanmark points on a face
    :param x_range: int, Max X of the image
    :return: 2D Int Array, Regression Line
    """
    regress = LinearRegression()
    regress_x = np.array([[x] for x in landmarks[:, 0]])
    regress_y = np.array(landmarks[:, 2])
    regress.fit(regress_x, regress_y)
    regress_line = regress.predict([[x] for x in range(x_range)])
#    print(regress.coef_, regress.intercept_, np.arctan(regress.coef_[0]))

    return regress_line, regress


def generate3d_regression(landmarks, x_range):
    """
    Generates a regression line of the landmarks on a face

    :param landmarks: 3D NP Array, representing lanmark points on a face
    :param x_range: int, Max X of the image
    :return: 2D Int Array, Regression Line
    """
    print(x_range)
    regress = LinearRegression()
    regress_x = np.array([[lm[0], lm[1]] for lm in landmarks])
    regress_y = np.array(landmarks[:, 2])
    regress.fit(regress_x, regress_y)
    print(regress.coef_, regress.intercept_, np.arctan(regress.coef_[0]))

    no_projection = np.array(regress.predict([[x, 0] for x in range(x_range)]))

    return no_projection


def lm_analysis(norm_landmarks, real_landmarks, lr_line):
    """
    Finds if points in a landmark array are on the top half

    :param norm_landmarks: 3d array, normalized face landmarks
    :param real_landmarks: 3d array, real face landmarks
    :param lr_line: 2d array, linear regression output
    :return:
    """
    lm_data = []
    for lm_idx in range(len(real_landmarks)):
        if lr_line[int(real_landmarks[lm_idx][0])] < real_landmarks[lm_idx][2]:
            lm_data.append([True, lm_idx, norm_landmarks[lm_idx]])
        else:
            lm_data.append([False, lm_idx, norm_landmarks[lm_idx]])

    return lm_data


def remove_image_lower_half(image, image_max_x, image_max_y, lr_line):
    """
    Takes an image and removes lower half based on linear regression model

    :param image: cv2 image, image to be edited
    :param image_max_x: int, max x of image
    :param image_max_y: int, max y of image
    :param lr_line: 2D int array, linear regression model
    :return: cv2 image with lower half removed
    """
    new_image = image.copy()
    for x in range(image_max_x):
        y = int(lr_line[x])
        while y < image_max_y:
            new_image[y, x] = [0, 0, 0]
            y += 1
    return new_image


def remove_image_upper_half(image, image_max_x, lr_line):
    """
    Takes an image and removes lower half based on linear regression model

    :param image: cv2 image, image to be edited
    :param image_max_x: int, max x of image
    :param lr_line: 2D int array, linear regression model
    :return: cv2 image with lower half removed
    """
    new_image = image.copy()
    for x in range(image_max_x):
        y = int(lr_line[x])
        while y >= 0:
            new_image[y, x] = [0, 0, 0]
            y -= 1
    return new_image


def plot_data(face_img, mesh_img, uh_image, lh_image, real_lms, lr_line, stan_landmarks):
    """Plots the data generated by single face"""
    plt.figure(figsize=(20, 5))
    ax1 = plt.subplot(161)
    ax2 = plt.subplot(162)
    ax3 = plt.subplot(163, projection='3d')
    ax4 = plt.subplot(164)
    ax5 = plt.subplot(165, projection='3d')
    ax6 = plt.subplot(166)
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax3.view_init(90, 270)
    ax4.axis("off")
    ax5.view_init(90, 270)
    plt.tight_layout()
    ax1.imshow(face_img)
    ax2.imshow(mesh_img)
    ax3.scatter(real_lms[:, 0],
                face_img.shape[0] - real_lms[:, 2],
                real_lms[:, 1])
    ax4.imshow(face_img)
    ax4.scatter([x[0] for x in real_lms],
                [y[2] for y in real_lms], s=4)
    ax4.plot([x for x in range(face_img.shape[1])], lr_line)
    ax5.scatter(stan_landmarks[:, 0],
                stan_landmarks[:, 2],
                stan_landmarks[:, 1])
    ax6.scatter(stan_landmarks[:, 0],
                stan_landmarks[:, 2])
    plt.show()


def plot_2dv3d(face_img, real_lms, lr, lr2):
    plt.figure()
    ax1 = plt.subplot(111)
    ax1.imshow(face_img)
    ax1.scatter([x[0] for x in real_lms],
                [y[2] for y in real_lms], s=4)
    ax1.plot([x for x in range(face_img.shape[1])], lr)
    ax1.plot([x for x in range(face_img.shape[1])], lr2)
    plt.show()


def make_nds_files():
    with open(UH_CSV, 'w') as uh_nds, \
            open(LH_CSV, 'w') as lh_nds,\
            open(ALL_CSV, 'w') as all_nds:
        uh_nds.write("fname\n")
        lh_nds.write("fname\n")
        all_nds.write("fname\n")
    with open(os.path.join(DATASET_ROOT, DATASET_INPUTS, 'normed.csv'), 'w') as nm:
        nm.write("fname\n")


if __name__ == "__main__":
#    make_nds_files()
    files_of_interest = ["133573.jpg", "004886.jpg", "005393.jpg", "006160.jpg", "133834.jpg"]
#    above = []
#    below = []
#    all_data = []
#    for i in files_of_interest:
#         generate_face_data(FULL_CELEB_IMAGES + i, True, False, False)
#        above.append(a)
#        below.append(b)

    generate_face_data("cruboker.jpg", True)
    generate_face_data("looking_down.jpg", True)
    generate_face_data(os.path.join(FULL_CELEB_IMAGES, '000001.jpg'), True)

#    for i in range(1, 100):
#        generate_face_data(FULL_CELEB_IMAGES + f'{i:>06d}.jpg', True, False, False)
#    for i in range(1, 202599):
#        generate_face_data(os.path.join(FULL_CELEB_IMAGES, f'{i:>06d}.jpg'), False, False, False)
#        except TypeError:
#            pass
#        above.append(a)
#        below.append(b)
#        all_data.append(c)
#
#    all_tensor = torch.tensor(all_data)
#    torch.corrcoef(all_tensor)
#    plt.imshow(all_tensor, cmap='BuPu', interpolation='nearest')
#    plt.show()
#    perfect = [116, 181, 250, 273, 293, 320, 361, 363, 472, 520, 568, 612, 657, 659, 697, 724, 791, 807, 809, 816, 817, 819, 835, 904]
#    for i in perfect:
#        a,b,c = generate_face_data(CELEBA_DATA + f'{i:>06d}.jpg', True, True, True)
#
#        above.append(a)
#        below.append(b)
#        all_data.append(c)


    # for i in range(1, 202599):
#       generate_face_data(CELEBA_DATA + f'{i:>06d}.jpg', i % 1000, )
    # TODO: MOVE TO A NOTEBOOK
#    an_lms = lm_analysis(norm_landmarks, real_landmarks, lr_line)
#    for i in range(len(an_lms)):
#        # print(an_lms[i])
#        print(i, an_lms[i][0], an_lms[i][2])
#    plt.imshow(image)
#    for i in an_lms:
#        plt.plot(i[0], i[2])
#        plt.draw()
#        plt.pause(.1)
#        plt.clf()
#    plt.show()

#    an_lms = np.array(an_lms)
#    x_avg = np.average(an_lms[:, 0])
#    y_avg = np.average(an_lms[:, 2])
#    z_avg = np.average(an_lms[:, 1])
#    print(x_avg, y_avg, z_avg)
#    plt.figure()
#    ax1 = plt.subplot(611)
#    ax2 = plt.subplot(612)
#    ax3 = plt.subplot(613)
#    ax4 = plt.subplot(614)
#    ax5 = plt.subplot(615, projection='3d')
#    ax6 = plt.subplot(616)
#    ax1.plot([i for i in range(len(an_lms))], [j[0] for j in an_lms])
#    ax1.plot([i for i in range(len(an_lms))], [x_avg for j in an_lms])
#    ax2.plot([i for i in range(len(an_lms))], [j[2] for j in an_lms])
#    ax2.plot([i for i in range(len(an_lms))], [y_avg for j in an_lms])
#    ax3.plot([i for i in range(len(an_lms))], [j[1] for j in an_lms])
#    ax3.plot([i for i in range(len(an_lms))], [z_avg for j in an_lms])
#    ax4.plot([i for i in range(len(an_lms))], [np.sqrt(j[0] ** 2 + j[1] ** 2 + j[2] ** 2) for j in an_lms])
#  ax5.plot([i for i in range(len(an_lms))], [j[0] for j in an_lms], [k[2] for k in an_lms])
#    ax6.plot([i for i in range(len(an_lms))],
#             [np.sqrt(((j[0] - x_avg) ** 2 + (j[1] - z_avg) ** 2 + (j[2] - y_avg) ** 2)) for j in an_lms])
#    plt.show()
#    plt.imshow(image)
#    ran = [10,11]
#    for i in ran:
#        plt.plot(real_landmarks[i][0], real_landmarks[i][2], marker='o', markersize=2)
#        plt.annotate(i, (real_landmarks[i][0], real_landmarks[i][2]))
#    plt.plot(real_landmarks[ran, 0], real_landmarks[ran, 2], '--')
#    plt.show()
#
#    print()
#    print()
#
#    plot = False
