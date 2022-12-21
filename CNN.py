# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization
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
import imgaug as ia
import imgaug.augmenters as iaa
import sklearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import skimage 
from skimage.color import rgba2rgb
import imgaug as ia
import imgaug.augmenters as iaa
from tensorflow.keras.models import load_model
import math

print(tf.__version__)
configproto = tf.compat.v1.ConfigProto() 
configproto.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=configproto) 
tf.compat.v1.keras.backend.set_session(sess)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) 
    return lr


dim=256


inputfile='dataset'+str(dim)+'.pickle'
infile = open(inputfile,'rb')
data = pickle.load(infile)
infile.close()
images = [i[0] for i in data]
print(str(type(images)))
#pick only the 2nd half of the images
# images = images[len(images)//2:]

coords = [i[1] for i in data]

# plt.show()
# dim=120
# for i in range(len(images)):
#     plt.clf()
#     img=images[i]
#     width=img.shape[1]
#     height=img.shape[0] 

#     img=skimage.transform.resize(img,(dim,dim))
#     plt.imshow(img,cmap='gray')

#     coordsss=coords[i]
#     plt.scatter(coordsss[0][0]*dim/width,coordsss[0][1]*dim/height,color='red',s=5)
#     plt.scatter(coordsss[1][0]*dim/width,coordsss[1][1]*dim/height,color='red',s=5)
#     plt.scatter(coordsss[2][0]*dim/width,coordsss[2][1]*dim/height,color='red',s=5)
#     plt.scatter(coordsss[3][0]*dim/width,coordsss[3][1]*dim/height,color='red',s=5)
    

#     plt.pause(0.3)


#pick only the 2nd half of the coords
# coords = coords[len(coords)//2:]



# #merge the 8 coordinates into one list
coords_list=[]
for i in range(len(coords)):
    coords_merged = []
    coords_merged.append(coords[i][0][0])
    coords_merged.append(coords[i][0][1])
    coords_merged.append(coords[i][1][0])
    coords_merged.append(coords[i][1][1])
    coords_merged.append(coords[i][2][0])
    coords_merged.append(coords[i][2][1])
    coords_merged.append(coords[i][3][0])
    coords_merged.append(coords[i][3][1])
    coords_list.append(coords_merged)

coords=coords_list

#convert images and coords to array
x = np.array(images)
y = np.array(coords)
#divide the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape,x_test.shape, y_test.shape)

#print the dtype of dataset
print("x_train: "+str(x_train.dtype))
print("y_train: "+str(y_train.dtype))
print("x_test: "+str(x_test.dtype))
print("y_test: "+str(y_test.dtype))




# %%
#CREATE THE MODEL
num_classes = 8

#retrieve  shape of x_train
input_shape = x_train.shape[1:]
#append 1
input_shape = input_shape + (1,)


model_1 = Sequential()

#input block
model_1.add(Conv2D(8, (3,3), strides = (1,1), padding='same',activation='relu',input_shape=input_shape))
model_1.add(Conv2D(8, (3,3), strides = (1,1), padding='same',activation='relu'))
#batch normalization
model_1.add(BatchNormalization())
model_1.add(MaxPooling2D(pool_size=(2,2)))


# block 2
model_1.add(Conv2D(16, (3,3), strides = (1,1), padding='same',activation='relu'))
model_1.add(Conv2D(16, (3,3), strides = (1,1), padding='same',activation='relu'))

#batch normalization
model_1.add(BatchNormalization())
model_1.add(MaxPooling2D(pool_size=(2, 2)))


# block 3
model_1.add(Conv2D(32, (3,3), strides = (1,1), padding='same',activation='relu'))
model_1.add(Conv2D(32, (3,3), strides = (1,1), padding='same',activation='relu'))

#batch normalization
model_1.add(BatchNormalization())
model_1.add(MaxPooling2D(pool_size=(2, 2)))


# block 4
model_1.add(Conv2D(64, (3,3), strides = (1,1), padding='same',activation='relu'))
model_1.add(Conv2D(64, (3,3), strides = (1,1), padding='same',activation='relu'))

#batch normalization
model_1.add(BatchNormalization())
model_1.add(MaxPooling2D(pool_size=(2, 2)))


# block 5
model_1.add(Conv2D(128, (3,3), strides = (1,1), padding='same',activation='relu'))
model_1.add(Conv2D(128, (3,3), strides = (1,1), padding='same',activation='relu'))

#batch normalization
model_1.add(BatchNormalization())
model_1.add(MaxPooling2D(pool_size=(2, 2)))






model_1.add(Flatten())
model_1.add(Dense(256))
model_1.add(Activation('relu'))

model_1.add(Dense(num_classes))


model_1.summary()

#%%
#LOAD A PRETRAINED MODEL
saved_name = 'saved'+str(dim)+' 5'
model_1=load_model(saved_name,compile=False)
model_1.compile(loss='mean_squared_error',
              metrics=['mae'])
# val_loss,val_mae = model_1.evaluate(x_test, y_test, verbose=0)
# model_1.summary()
# print(f"Best loss in evaluation: {val_loss},mae: {val_mae}\n")
print("loaded model "+saved_name)



# %%
#INITIALIZE THE MODEL
batch_size = 180
print("batch_size: "+str(batch_size))

#early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)

# initiate RMSprop optimizer
#opt = RMSprop(learning_rate=0.0001, decay=1e-3,momentum=0.9)
opt = keras.optimizers.Adam(learning_rate=0.0005,decay=1e-3)
lr_metric = get_lr_metric(opt)

#print learning rate and decay
print(opt.get_config())

# %%
#TRAIN THE MODEL
model_1.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['mae',lr_metric])


run_hist_1=model_1.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=100,
              validation_data=(x_test, y_test),
              callbacks=[callback],
              shuffle=True)

#print how many epochs were run
print("terminato a epoca: ", len(run_hist_1.history['loss']))
print(run_hist_1.epoch)

best_loss = min(run_hist_1.history["loss"])
best_mae = min(run_hist_1.history["mae"])
val_loss,val_mae,lr = model_1.evaluate(x_test, y_test, verbose=0)
print(f"Best loss in training: {best_loss}. In evaluation: {val_loss}")
print(f"Best mae in training: {best_mae}. In evaluation: {val_mae}")




# %%
fig, ax = plt.subplots()
ax.plot(run_hist_1.history["loss"],'r', marker='.', label="Train Loss")
ax.plot(run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss")
ax.legend()

fig, ax = plt.subplots()
ax.plot(run_hist_1.history["mae"],'g', marker='.', label="Train mae")
ax.plot(run_hist_1.history["val_mae"],'k', marker='.', label="Validation mae")
ax.legend()




# %%
###############################################################################
###############################################################################
###############################################################################
#SAVE MODEL
outputfilename='saved'+str(dim)+'_5'
with open(outputfilename+'_history', 'wb') as file_pi:
    pickle.dump(run_hist_1.history, file_pi)
model_1.save(outputfilename)
###############################################################################
###############################################################################
###############################################################################




# %%
saved_name = 'saved'+str(dim)+'_5'
savedModel=load_model(saved_name,compile=False)
savedModel.compile(loss='mean_squared_error')

#calculate predictions
predictions = savedModel.predict(x_test)
images_test = x_test.tolist()
plt.rcParams["figure.figsize"] = (15,15)
while(1):
    plt.clf()
    #create a random variable to select the image
    rnd = np.random.randint(0,len(images_test))
    print("previsione di: "+str(rnd))

    #print randomly all iamges and their coordinates
    plt.imshow(images_test[rnd],cmap='gray')
    #scatter the coords
    plt.scatter(y_test[rnd][0],y_test[rnd][1],c='g',s=10)
    plt.scatter(y_test[rnd][2],y_test[rnd][3],c='g',s=10)
    plt.scatter(y_test[rnd][4],y_test[rnd][5],c='g',s=10)
    plt.scatter(y_test[rnd][6],y_test[rnd][7],c='g',s=10)
    #scatter the prediction
    plt.scatter(predictions[rnd][0],predictions[rnd][1],color='red',s=10)
    plt.scatter(predictions[rnd][2],predictions[rnd][3],color='red',s=10)
    plt.scatter(predictions[rnd][4],predictions[rnd][5],color='red',s=10)
    plt.scatter(predictions[rnd][6],predictions[rnd][7],color='red',s=10)
    

    #wait 1 sec and clear plot
    plt.pause(1)


# %%
