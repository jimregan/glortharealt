#!/usr/bin/env python
import sys
import os
import dlib
import glob

predictor_path = 'shape_predictor_5_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
predictor2_path = 'shape_predictor_68_face_landmarks.dat
faces_folder_path = sys.argv[1]

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
sp2 = dlib.shape_predictor(predictor2_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

output = open('{}.txt'.format(faces_folder_path), 'w+')
output2 = open('{}.68.txt'.format(faces_folder_path), 'w+')
output_folder_path = '{}_out'.format(faces_folder_path)

descriptors = []
images = []

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
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    descriptors.append(face_descriptor)
    images.append((img, shape))

# Now let's cluster the faces.  
labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
num_classes = len(set(labels))
print("Number of clusters: {}".format(num_classes))

# Find biggest class
biggest_class = None
biggest_class_length = 0
for i in range(0, num_classes):
    class_length = len([label for label in labels if label == i])
    if class_length > biggest_class_length:
        biggest_class_length = class_length
        biggest_class = i

print("Biggest cluster id number: {}".format(biggest_class))
print("Number of faces in biggest cluster: {}".format(biggest_class_length))

# Find the indices for the biggest class
indices = []
for i, label in enumerate(labels):
    if label == biggest_class:
        indices.append(i)

print("Indices of images in the biggest cluster: {}".format(str(indices)))

# Ensure output directory exists
if not os.path.isdir(output_folder_path):
    os.makedirs(output_folder_path)

# Save the extracted faces
print("Saving faces in largest cluster to output folder...")
for i, index in enumerate(indices):
    img, shape = images[index]
    file_path = os.path.join(output_folder_path, "face_" + str(i))
    dlib.save_face_chip(img, shape, file_path)

output.close()
output2.close()