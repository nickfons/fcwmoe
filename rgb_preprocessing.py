"""
Pre-processing for Machine Learning algorithms dealing with the RGB Space

Created: 5/9/22 - NF
Modified: 5/9/22 - NF
"""
import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def align_and_crop(inp_fn, out_fn, plot=False):
    image = cv2.imread(inp_fn)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_x = image.shape[1]
    image_y = image.shape[0]

    try:
        face_landmarks = generate_mp_data(image)
    except TypeError as e:
        return
    landmarks = np.array([[landmark.x * image_x, landmark.y * image_y]
                            for landmark in face_landmarks.landmark])
    landmarks_z = np.array([landmark.y for landmark in face_landmarks.landmark])
    landmarks_z = (landmarks_z - np.min(landmarks_z)) / (np.max(landmarks_z) - np.min(landmarks_z)) * 254
    nose_point = landmarks[5]
    slope = (landmarks[454][1] - landmarks[234][1])/(landmarks[454][0] - landmarks[234][0])
    angle = np.arctan(slope)
    rot_mat = cv2.getRotationMatrix2D(nose_point, np.degrees(angle), 1.0)
    aligned_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    aligned_landmarks = rotate(landmarks, nose_point, - angle)
    face_shape = np.array([[np.min(aligned_landmarks[:, 0]), np.max(aligned_landmarks[:, 0])],
                           [np.min(aligned_landmarks[:, 1]), np.max(aligned_landmarks[:, 1])]])
#    print(face_shape)
    cropped_image = aligned_image[int(face_shape[1][0]):int(face_shape[1][1]),
                    int(face_shape[0][0]):int(face_shape[0][1])]
    cropped_landmarks = aligned_landmarks - [face_shape[0][0], face_shape[1][0]]
#    print((nose_point[1] - face_shape[1][0]) / (face_shape[1][1] - nose_point[1]))
    cropped_face_shape = np.array([[0, face_shape[0][1] - face_shape[0][0]],
                                   [0, face_shape[1][1] - face_shape[1][0]]])
#    print(cropped_face_shape)
    scaled_image = cv2.resize(cropped_image, [128, 128])
    scaled_landmarks = cropped_landmarks / np.array([[cropped_face_shape[0][1]],
                                                    [cropped_face_shape[1][1]]]).T * np.array([127, 127])
    was_flipped = False
    if scaled_landmarks[5][0] > 64:
        scaled_landmarks = np.array([[127 - pt[0], pt[1]] for pt in scaled_landmarks])
        scaled_image = cv2.flip(scaled_image, 1)
        was_flipped = True
    point_map = np.array([[[0, 0, 0] for _ in range(128)] for _ in range(128)])
    for idx, point in enumerate(scaled_landmarks):
        if point_map[int(point[1])][int(point[0])].any():
            point_map[int(point[1])][int(point[0])][1] = landmarks_z[idx]
        point_map[int(point[1])][int(point[0])] = [landmarks_z[idx], 255, 255]

    scaled_nose_pixel = int(scaled_landmarks[5][1])
    warp_image = np.concatenate((cv2.resize(scaled_image[:scaled_nose_pixel][0:127], (128, 64)),
                                 cv2.resize(scaled_image[scaled_nose_pixel:][0:127], (128, 64))))
    warp_point_map = np.array([[[0, 0, 0] for _ in range(128)] for _ in range(128)])
    for idx, org_point in enumerate(scaled_landmarks):
        if int(org_point[1]) <= scaled_nose_pixel:
            point = [org_point[0], org_point[1] * (64 / scaled_nose_pixel)]
        else:
            point = [org_point[0], 64 + 64 * ((org_point[1] - scaled_nose_pixel) / (128 - scaled_nose_pixel))]
        if warp_point_map[int(point[1])][int(point[0])].any():
            warp_point_map[int(point[1])][int(point[0])][1] = landmarks_z[idx]
        warp_point_map[int(point[1])][int(point[0])] = [landmarks_z[idx], 255, 255]

    if plot:
        plt.figure()
        ax1 = plt.subplot(181)
        ax2 = plt.subplot(182)
        ax3 = plt.subplot(183)
        ax4 = plt.subplot(184)
        ax5 = plt.subplot(185)
        ax6 = plt.subplot(186)
        ax7 = plt.subplot(187)
        ax8 = plt.subplot(188)
        ax1.set_aspect(1)
        ax2.set_aspect(1)
        ax3.set_aspect(1)
        ax4.set_aspect(1)
        ax5.set_aspect(1)
        ax6.set_aspect(1)
        ax7.set_aspect(1)
        ax8.set_aspect(1)
#        ax1.set_title('Original')
#        ax2.set_title('Landmarks')
#        ax3.set_title('Aligned')
#        ax4.set_title('Cropped')
#        ax5.set_title('Scaled/Flipped')
#        ax6.set_title('SF Landmarks')
#        ax7.set_title('Warped')
#        ax8.set_title('Warped Landmarks')
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])
        ax4.set_xticklabels([])
        ax5.set_xticklabels([])
        ax6.set_xticklabels([])
        ax7.set_xticklabels([])
        ax8.set_xticklabels([])
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax4.set_yticklabels([])
        ax5.set_yticklabels([])
        ax6.set_yticklabels([])
        ax7.set_yticklabels([])
        ax8.set_yticklabels([])
        ax1.imshow(image)
        ax2.scatter(landmarks[:, 0], image_y - landmarks[:, 1])
        ax3.imshow(aligned_image)
        str_line = np.array([nose_point[1] for x in range(image_x)])
        ax3.plot(str_line)
        ax4.imshow(cropped_image)
        ax4.plot(np.array([cropped_landmarks[5][1] for x in range(cropped_image.shape[1])]))
        ax5.imshow(scaled_image)
        str_line = np.array([scaled_nose_pixel for x in range(128)])
        mid_line = np.array([64 for x in range(128)])
        ax5.plot(str_line)
        ax5.plot(mid_line)
        ax6.plot(str_line)
        ax6.plot(mid_line)
        ax6.imshow(point_map)
        ax7.imshow(warp_image)
        ax8.imshow(warp_point_map)
        ax7.plot(str_line, mid_line)
        ax7.plot(mid_line)
        ax8.plot(str_line, mid_line)
        ax8.plot(mid_line)

        plt.show()

    scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_RGB2BGR)
    warp_image = cv2.cvtColor(warp_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_fn, "cropped", inp_fn[-10:]), scaled_image)
    cv2.imwrite(os.path.join(out_fn, "dot", inp_fn[-10:]), point_map)
    cv2.imwrite(os.path.join(out_fn, "warp", inp_fn[-10:]), warp_image)
    cv2.imwrite(os.path.join(out_fn, "warp_dot", inp_fn[-10:]), warp_point_map)

    return face_shape, scaled_nose_pixel, was_flipped


def generate_mp_data(image_fp):
    """
    Generates a 3D array of face landmark points by analyzing a given
    image with Google's Mediapipe Library

    :param image_fp: cv2 image variable
    :return: landmarks: 2D Numpy array of real landmarks
    """
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False,
                                            min_detection_confidence=.5) as face_mesh:
        multi_face = face_mesh.process(image_fp)
        try:
            face_landmarks = multi_face.multi_face_landmarks[0]
        except TypeError as e:
            raise e

        return face_landmarks


def rotate(array, mid_point, angle):
    p_mat = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]
                     ])
    mid_point = np.array([mid_point])
    return (p_mat @ (array - mid_point).T).T + mid_point


if __name__ == "__main__":
#    align_and_crop("cruboker.jpg", '')
    real_names = []

#    align_and_crop(f"C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\all\\000881.jpg",
#                    'C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\')
#    align_and_crop(f"C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\all\\000080.jpg",
#               'C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\')
#    align_and_crop(f"C:\\Users\\nicho\programming\\483project\\looking_down.jpg",
#               'C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\')
#    align_and_crop(f"C:\\Users\\nicho\programming\\483project\\cruboker.jpg",
#               'C:\\Users\\nicho\programming\\483project\\rand\\', True)
#    align_and_crop(f"C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\all\\000167.jpg",
#               'C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\', True)
#    align_and_crop(f"C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\all\\000168.jpg",
#                   'C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\', True)
#    align_and_crop(f"C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\all\\000169.jpg",
#               'C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\', True)
#    align_and_crop(f"C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\all\\000547.jpg",
#               'C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\', True)
#    for i in range(1, 202599):
#    with open(os.path.join('dataset', 'outputs', 'rgb_info.csv'), 'w') as fp:
#        fp.write('fname,fsx1,fsx2,fsy1,fsy2,nx,ny,flipped')

#        for i in range(1, 202599):
#        for i in range(1, 100):
#            try:
#                face_shape, nose_pixel, flipped = align_and_crop(f"C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\all\\{i:>06d}.jpg", 'C:\\Users\\nicho\programming\\483project\\dataset\\inputs\\')
#            except:
#                pass
#            else:
#                fp.write(f'{i:>06d},{face_shape[0][0]:.6f},{face_shape[1][0]:.6f},{face_shape[0][1]:.6f},{face_shape[1][1]:.6f},{nose_pixel},{flipped}')
#                real_names.append(i)
