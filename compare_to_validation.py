__author__ = 'Iacopo'
import renderer
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
import check_resources as check
import matplotlib.pyplot as plt
import sys
import myutil
import ThreeD_Model
import config
from camera_calibration import get_yaw_pitch_roll


def preprocess_config():
    global this_path, opts, pose_models_folder, pose_models
    this_path = os.path.dirname(os.path.abspath(__file__))
    opts = config.parse_for_validation()
    # 3D Models we are gonna use to to the rendering {0, -40, -75}
    newModels = opts.getboolean('renderer', 'newRenderedViews')
    pose_models_folder = '/model_lecturer/'
    if opts.getboolean('renderer', 'newRenderedViews'):
        pose_models = ['model3D_aug_-00_00', ]
    else:
        assert False
        pose_models = ['model3D_aug_-00', 'model3D_aug_-40', 'model3D_aug_-75']
    # In case we want to crop the final image for each pose specified above/
    # Each bbox should be [tlx,tly,brx,bry]
    resizeCNN = opts.getboolean('general', 'resizeCNN')
    cnnSize = opts.getint('general', 'cnnSize')
    if not opts.getboolean('general', 'resnetON'):
        crop_models = [None, None, None, None, None]  # <-- with this no crop is done.
    else:
        # In case we want to produce images for ResNet
        resizeCNN = False  # We can decide to resize it later using the CNN software or now here.
        # The images produced without resizing could be useful to provide a reference system for in-plane alignment
        cnnSize = 224
        crop_models = [[23, 0, 23 + 125, 160], [0, 0, 210, 230], [0, 0, 210, 230]]  # <-- best crop for ResNet


def create_paths():
    path_to_images_folder = r"C:\Noam\Code\vision_course\face_pose_estimation\images\valid_set\images"
    image_paths = []
    for file_path in os.listdir(path_to_images_folder):
        if file_path.endswith(".png"):
            image_paths.append(os.path.join(path_to_images_folder, file_path))
    print(f"#images: {len(image_paths)}")

    points_paths = []
    for file_path in os.listdir(path_to_images_folder):
        if file_path.endswith(".pts"):
            points_paths .append(os.path.join(path_to_images_folder, file_path))
    print(f"#points: {len(points_paths )}")

    # cull images that don't have points
    images_paths_with_points = []
    for image_path in image_paths:
        for points_path in points_paths:
            if image_path.split(".png")[0] == points_path.split(".pts")[0]:
                images_paths_with_points.append(image_path)
                break

    # cull pointa that don't have images
    points_paths_with_images = []
    for points_path in points_paths:
        for image_path in image_paths:
            if image_path.split(".png")[0] != points_path.split(".pts")[0]:
                points_paths_with_images.append(points_path)
                break

    assert len(images_paths_with_points) == len(points_paths_with_images)
    print(f"#tagged images = {len(images_paths_with_points)}")

    return sorted(images_paths_with_points), sorted(points_paths_with_images)


def demo():
    preprocess_config()
    n_sub = opts.getint('general', 'nTotSub')
    paths = create_paths()
    path_to_image = sys.argv
    file_list, output_folder = myutil.parse(path_to_image)

    # check for dlib saved weights for face landmark detection
    # if it fails, dowload and extract it manually from
    # http://sourceforge.net/projects/dclib/files/d.10/shape_predictor_68_face_landmarks.dat.bz2
    check.check_dlib_landmark_weights()

    # Pre loading all the models for speed
    all_models = myutil.preload(this_path, pose_models_folder, pose_models, n_sub)

    for f in file_list:
        if '#' in f:  # skipping comments
            continue
        splitted = f.split(',')
        image_key = splitted[0]
        image_path = splitted[1]
        image_landmarks = splitted[2]
        img = cv2.imread(image_path, 1)
        if image_landmarks != "None":
            landmark = np.loadtxt(image_landmarks)
            landmarks = list()
            landmarks.append(landmark)
        else:
            print('> Detecting landmarks')
            landmarks = feature_detection.get_landmarks(img, this_path)

        if len(landmarks) != 0:
            # Copy back original image and flipping image in case we need
            # This flipping is performed using all the model or all the poses
            # To refine the estimation of yaw. Yaw can change from model to model...

            img_display = img.copy()
            img, landmarks, img_yaw = myutil.flipInCase(img, landmarks, all_models)
            # listPose = myutil.decidePose(yaw, opts, newModels)
            list_pose = [0]
            # Looping over the poses
            # for poseId in listPose:
            for poseId in list_pose:
                posee = pose_models[poseId]
                # Looping over the subjects
                for subj in range(1, n_sub + 1):
                    pose = posee + '_' + str(subj).zfill(2) + '.mat'
                    print('> Looking at file: ' + image_path + ' with ' + pose)
                    # load detections performed by dlib library on 3D model and Reference Image
                    print("> Using pose model in " + pose)
                    # Indexing the right model instead of loading it each time from memory.
                    model3D = all_models[pose]
                    eye_mask = model3D.eyemask
                    # perform camera calibration according to the first face detected
                    proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, landmarks[0])
                    yaw, pitch, roll = get_yaw_pitch_roll(rmat)
                    print(f"yaw = {yaw}, pitch={pitch}, roll={roll}")


if __name__ == "__main__":
    demo()
