# CNN in tensorflow for regression(4 points)

### The problem
Suppose you have some images with associated txt files in the same directory like:

    ├───DB
        ├───1.jpg
        ├───1.txt
        ├───2.jpg
        ├───2.txt

and in each file you have the coords of the points (here we have 4) like this:

501,203
803,708
...(x,y one line for each point)

## DATA AUGMENTATION
With data_augmentation.py you can read the the images and files, apply some transformation to increase the dataset
and finally save it as .pickle.

Remember to change the '\path\to\db' with your db folder

## CNN

Cnn.py contains the model, with related training.


If you have answers fell free to ask!
