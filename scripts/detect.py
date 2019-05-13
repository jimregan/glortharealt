#!/usr/bin/env python
import sys
import os
import dlib
import glob
import time

start = time.time()

predictor_path = 'shape_predictor_5_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
predictor2_path = 'shape_predictor_68_face_landmarks.dat'
faces_folder_path = sys.argv[1]

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
sp2 = dlib.shape_predictor(predictor2_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

output = open('{}.txt'.format(faces_folder_path), 'w+')
output2 = open('{}.68.txt'.format(faces_folder_path), 'w+')
output3 = open('{}.id.txt'.format(faces_folder_path), 'w+')
output_folder = '{}_out'.format(faces_folder_path)

descriptors = []
images = []
regions = []
ridx = 0

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
    base_region = "{}\t{}\t{}\t{}\t{}\t{}".format(f, k, d.left(), d.top(), d.right(), d.bottom())
    output2.write("{}\t{}\n".format(base_region, ' '.join(format_p68(shape))))
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    descriptors.append(face_descriptor)
    images.append((img, shape))
    regions.append(ridx)
    ridx = ridx + 1

# Now let's cluster the faces.  
labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
num_classes = len(set(labels))
print("Number of clusters: {}".format(num_classes))

for i in range(0, num_classes):
    indices = []
    class_length = len([label for label in labels if label == i])
    for j, label in enumerate(labels):
        if label == i:
            indices.append(j)
    print("Indices of images in the cluster {0} : {1}".format(str(i),str(indices)))
    print("Size of cluster {0} : {1}".format(str(i),str(class_length)))
    output_folder_path = output_folder + '/output' + str(i) # Output folder for each cluster
    os.path.normpath(output_folder_path)
    os.makedirs(output_folder_path)
    
    # Save each face to the respective cluster folder
    print("Saving faces to output folder...")
    for k, index in enumerate(indices):
        img, shape = images[index]
        file_path = os.path.join(output_folder_path, "face_idx"+str(regions[index])+"_"+str(k)+"_"+str(i))
        dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)
        output3.write("idx{}\tid{}\n".format(str(regions[index]), str(i)))

output.close()
output2.close()
output3.close()
