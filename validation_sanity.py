import pandas as pd
import numpy as np
import os
import dlib
import ThreeD_Model


def preload(this_path, pose_models_folder, pose_models, nSub):
    print('> Preloading all the models for efficiency')
    allModels = dict()
    for posee in pose_models:

        # Looping over the subjects
        for subj in range(1, nSub + 1):
            pose = posee + '_' + str(subj).zfill(2) + '.mat'
            # load detections performed by dlib library on 3D model and Reference Image
            print("> Loading pose model in " + pose)
            # model3D = ThreeD_Model.FaceModel(this_path + "/models3d_new/" + pose, 'model3D')
            if '-00' in posee:
                model3d = ThreeD_Model.FaceModel(this_path + pose_models_folder + pose, 'model3D', True)
            else:
                model3d = ThreeD_Model.FaceModel(this_path + pose_models_folder + pose, 'model3D', False)

            allModels[pose] = model3d
    return allModels


def read_ground_truth_validation():
    ground_truth_file_path = r"C:\Noam\Code\vision_course\face_pose_estimation\images\valid_set\validation_set.csv"
    ground_truth = pd.read_csv(ground_truth_file_path,
                               sep=r'\s*,\s*',
                               header=0,
                               encoding='ascii',
                               engine='python')

    return ground_truth


def load_landmarks_from_path(dir_path, file_name):
    assert os.path.isdir(dir_path)
    file_path = os.path.abspath(os.path.join(dir_path, file_name))
    assert os.path.isfile(file_path)

    points = np.loadtxt(file_path, comments=("version:", "n_points:", "{", "}"))
    return points


if __name__ == "__main__":
    def main():
        validation_df = read_ground_truth_validation()
        validation_images_filenames = [filename for filename in validation_df['file name'].values]
        validation_pts_filenames = [filename.replace('.png', '.pts') for filename in validation_images_filenames]

        validation_images_folder_path = r"C:\Noam\Code\vision_course\face_pose_estimation\images\valid_set\images"
        validation_points = [load_landmarks_from_path(validation_images_folder_path, file_name) for file_name in validation_pts_filenames]

        # initialize dlib's face detector and create facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor_path = r"C:\Noam\Code\vision_course\shape_predictor\shape_predictor_68_face_landmarks.dat"
        assert os.path.isfile(predictor_path)
        predictor = dlib.shape_predictor(predictor_path)

        result = []
        this_path = r"C:\\Noam\\Code\\vision_course\\face_specific_augm"
        pose_models_folder = r'/model_lecturer/'
        pose_models = [r'model3D_aug_-00_00']
        n_sub = 1
        pose = r'model3D_aug_-00_00_01.mat'
        model3d = preload(this_path, pose_models_folder, pose_models, n_sub)[pose]




        hi = 5


    main()