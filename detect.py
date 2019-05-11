import sys
import os
import dlib
import glob

predictor_path = 'shape_predictor_5_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
predictor2_path = 'shape_predictor_68_face_landmarks.dat'
faces_folder_path = sys.argv[0]

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
sp2 = dlib.shape_predictor(predictor2_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

output = open('{}.txt'.format(faces_folder_path), 'w+')
output2 = open('{}.68.txt'.format(faces_folder_path), 'w+')

def format_p68(shape):
  points = list()
  for i in range(0, 68):
    points.append("{},{}".format(shape.part(i).x, shape.part(i).y))
  return points

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
  img = dlib.load_rgb_image(f)
  dets = detector(img, 1)
  
  for k, d in enumerate(dets):
    output.write("File: {} Detection {}: Left: {} Top: {} Right: {} Bottom: {}\n".format(
            f, k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp2(img, d)
    output2.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(f, k, d.left(), d.top(), d.right(), d.bottom(), ' '.join(format_p68(shape))))
    
output.close()