__author__ = 'Douglas'

import urllib.request, urllib.error, urllib.parse, os, bz2
dlib_facial_landmark_model_url = "http://ufpr.dl.sourceforge.net/project/dclib/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2"
import os


def download_file(url, dest):
    file_name = url.split('/')[-1]
    u = urllib.request.urlopen(url)
    f = open(dest+"/"+file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print("Downloading: %s Size: %s (~%4.2fMB)" % (file_name, file_size, (file_size/1024./1024.)))

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        file_size_dl += len(buffer)
        f.write(buffer)
        if((file_size_dl*100./file_size) % 5 <= 0.01):
            status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print(status)
    f.close()
    print("Download complete!")

def extract_bz2(fpath):
    print("Extracting...")
    new_file = open(fpath[:-4], "wb")
    file = bz2.BZ2File(fpath, 'rb')
    data = file.read()
    new_file.write(data)
    new_file.close()
    print("Done!")


def check_dlib_landmark_weights():
    dlib_models_folder = r"C:\Noam\Code\vision_course\shape_predictor"
    assert os.path.isdir(dlib_models_folder)
    predictor_file_path = os.path.join(dlib_models_folder, r"shape_predictor_68_face_landmarks.dat")
    assert os.path.isfile(predictor_file_path)
    # if not os.path.isfile(dlib_models_folder+"/shape_predictor_68_face_landmarks.dat.bz2"):
    #     download_file(dlib_facial_landmark_model_url, dlib_models_folder)
    # extract_bz2(dlib_models_folder+"/shape_predictor_68_face_landmarks.dat.bz2")
