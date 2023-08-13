import cv2
from os.path import exists
from os import mkdir
from sklearn.datasets import fetch_olivetti_faces

data, targets = fetch_olivetti_faces(return_X_y=True)

FOLDER = "olivetti_faces"
img_count = {}

if not exists(FOLDER):
  mkdir(FOLDER)
  
labels = np.unique(targets)

for i in range(labels.shape[0]):
  path = FOLDER+"/"+str(labels[i])
  if not exists(path):
    mkdir(path)

for i in range(data.shape[0]):
  label = targets[i]
  path = FOLDER+"/"+str(label)

  if label in img_count:
    img_count[label]+=1
  else:
    img_count[label] = 0

  cv2.imwrite(path+"/"+str(img_count[label])+".jpg", data[i].reshape(img_size[0], img_size[1])*255)

!zip -r olivetti_faces.zip olivetti_faces/ 
