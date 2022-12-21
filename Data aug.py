import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
import tqdm
import json
import pickle
import PIL
import skimage 
from skimage.color import rgba2rgb
import imgaug as ia
import imgaug.augmenters as iaa

# load all images in a directory
ImagesList = glob.glob('C:\\Users\\fogli\\OneDrive - unife.it\\Desktop\\UNI\\Deep Learning\\Progetto\\DB\\*.jpg')
loaded_data = list()
print("Loading Images...")
for filename in tqdm.tqdm(ImagesList,desc='Loading Images'):
	# load image
	img = mpimg.imread(filename)

	#if image have 4 cahnnels, convert to RGB
	if img.shape[2]==4:
		img=rgba2rgb(img)

	#convert to grayscale
	img = skimage.color.rgb2gray(img)
	name=os.path.basename(filename)

    #remove last 4 items from the name
	name = name[:-4]
	txt_file=glob.glob('C:\\Users\\fogli\\OneDrive - unife.it\\Desktop\\UNI\\Deep Learning\\Progetto\\DB\\'+name+'.txt')

	#read the data file
	with open(txt_file[0]) as f:
		lines=f.readlines()


	coord1=(float(lines[1].split(',')[0]),float(lines[1].split(',')[1]))
	coord2=(float(lines[2].split(',')[0]),float(lines[2].split(',')[1]))
	coord3=(float(lines[3].split(',')[0]),float(lines[3].split(',')[1]))
	coord4=(float(lines[4].split(',')[0]),float(lines[4].split(',')[1]))
	coords=[coord1,coord2,coord3,coord4]

	#save original width and height
	width=img.shape[1]
	height=img.shape[0]
	#resize the image
	dim=256
	img = skimage.transform.resize(img,(dim,dim))
	
	#resize coordinates
	for i in range(len(coords)):
		coords[i]=(coords[i][0]*dim/width,coords[i][1]*dim/height)

	data=(img,coords,name)
	loaded_data.append(data)

images = [i[0] for i in loaded_data]
coords = [i[1] for i in loaded_data]


#Add blur and linear contrast to the images
seq = iaa.Sequential([iaa.LinearContrast((0.6, 1.5)), iaa.GaussianBlur(sigma=(0., 0.0))])
images_aug_blur, coords_aug_blur = seq(images=images, keypoints=coords)
#add augmented images to the dataset
images.extend(images_aug_blur)
coords.extend(coords_aug_blur)

#flip the images
seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5)])
images_aug_flip, coords_aug_flip = seq(images=images, keypoints=coords)
#add augmented images to the dataset
images.extend(images_aug_flip)
coords.extend(coords_aug_flip)

#rotate and scale the images
seq = iaa.Sequential([iaa.Affine(rotate=(-20, 20)), iaa.Affine(scale=(0.8, 1))])
images_aug_rotate, coords_aug_rotate = seq(images=images, keypoints=coords)
#add augmented images to the dataset 
images.extend(images_aug_rotate)
coords.extend(coords_aug_rotate)

#rotate and scale the images again
seq = iaa.Sequential([iaa.Affine(rotate=(-20, 20)), iaa.Affine(scale=(0.8, 1))])
images_aug_rotate, coords_aug_rotate = seq(images=images, keypoints=coords)
#add augmented images to the dataset 
images.extend(images_aug_rotate)
coords.extend(coords_aug_rotate)

dataset=list(zip(images, coords))
#take only the first 20000 images
dataset=dataset[:20000]
#create a name with data + dim
name='dataset'+str(dim)+'.pickle'

#save dataset to file
outfile = open(name,'wb')
pickle.dump(dataset,outfile)
outfile.close()
	




