import argparse
import cv2
import os
import csv
from PIL import Image
import pathlib
import shutil 
def split_image(img):

        im = Image.open(img)
        w, h = im.size
        im1 = im.crop((0, 0, w // 2, h))
        im2 = im.crop((w // 2, 0, 2*w // 2, h))

        return im1,im2

def split_train_test(images_dir_in, images_dir_out):
    ''' Create output directories '''
    if not os.path.exists(images_dir_out + 'train/' ):
            os.makedirs(images_dir_out + 'train/' ) 
    if not os.path.exists(images_dir_out + 'val/' ):
            os.makedirs(images_dir_out + 'val/')


    ''' Loop over masks '''
    file_names = list(pathlib.Path(images_dir_in).rglob('*.png'))
    n_tot = len(file_names)
    n_train = int(n_tot*0.8)
    n_test = n_tot - n_train
    print('Number of training images: ', n_train)
    print('Number of test images: ', n_test)

    images_train = file_names[0:n_train]
    images_test = file_names[n_train:]


    for i,file_name in enumerate(images_train):
            shutil.move(images_dir_in + file_name.name, images_dir_out + 'train/'  + file_name.name)
    for i,file_name in enumerate(images_test):
           shutil.move(images_dir_in + file_name.name, images_dir_out + 'val/'  + file_name.name)


def get_immediate_filesNames(a_dir, ext:bool=1):
    if ext == 0:
        return [name.split('.')[0] for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name))]
    else:
        return [name for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name))]


def draw_rectangle(event, x, y, flags, param):
    global roi_pt, is_button_down
    global dataset_path, file_name
    if event == cv2.EVENT_MOUSEMOVE and is_button_down:
        global image_clone, image

        # get the original image to paint the new rectangle
        image = image_clone.copy()

        # draw new rectangle
        cv2.rectangle(image, roi_pt[0], (x,y), (0, 255, 0), 2)

    if event == cv2.EVENT_LBUTTONDOWN:
        # record the first point
        roi_pt = [(x, y)]  
        is_button_down = True

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:        
        roi_pt.append((x, y))     # append the end point
        
        # ======================
        # print the bounding box
        # ======================
        # in (x1,y1,x2,y2) format
        print(roi_pt)                  
        
        # in (x,y,w,h) format
        bbox = (roi_pt[0][0],
                roi_pt[0][1],
                roi_pt[1][0] - roi_pt[0][0],
                roi_pt[1][1] - roi_pt[0][1])

        f = open('Orientation_learning/Annotation/dataset/annotations.cvs', 'a')
        writer = csv.writer(f)
        writer.writerow([dataset_path + file_name ,str(roi_pt[0][0]) ,str(roi_pt[0][1]) ,str(roi_pt[1][0]) ,str(roi_pt[1][1]),'tool'])
        f.close()
        # button has now been released
        is_button_down = False

        # draw the bounding box
        cv2.rectangle(image, roi_pt[0], roi_pt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

if __name__ == "__main__": 

    divide_image = False
    split_dataset_train_test = False
    annotate = True

    if divide_image == True:
        dataset_raw_path = '/home/user/Orientation_learning/Annotation/dataset_raw/'
        dataset_path = '/home/user/Orientation_learning/Annotation/post_processed/'
        files = get_immediate_filesNames(dataset_raw_path)
        for i,file_name in enumerate(files):
            im1,im2 = split_image(dataset_raw_path + file_name)
            im1.save(dataset_path + str(i)+"_1.png")
            im2.save(dataset_path + str(i)+"_2.png")

    if split_dataset_train_test == True:

        input_dataset_path = '/home/user/Orientation_learning/Annotation/post_processed/'
        output_dataset_path = '/home/user/Orientation_learning/Annotation/dataset/'
        split_train_test( input_dataset_path, output_dataset_path)

    if annotate == True:
        dataset_path = '/home/user/Orientation_learning/Annotation/dataset/train/'
        files = get_immediate_filesNames(dataset_path)

        
        
        for i,file_name in enumerate(files):
            # to store the points for region of interest
            roi_pt = []
            # to indicate if the left mouse button is depressed
            is_button_down = False
            
            image = cv2.imread(dataset_path + file_name)

            # reference to the image
            image_clone = image
            # setup the mouse click handler
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", draw_rectangle)
            # loop until the 'q' key is pressed
            while True:
                # display the image 
                cv2.imshow("image", image)
                
                # wait for a keypress
                key = cv2.waitKey(1)
                if key == ord("c"):
                    break

            # close all open windows
            cv2.destroyAllWindows()
