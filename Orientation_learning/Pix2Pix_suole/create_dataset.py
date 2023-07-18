import cv2
import numpy as np
from PIL import Image
import random
import os
import pathlib
import shutil


def rectangular_occlusions(truth, path= '/home/user/Orientation_learning/Pix2Pix_suole/dataset_augmented_2/'):
        for nn in range(1,30):   # 2 immagini occluse con rettangoli per ogni ground truth

            # Random occlusions generation 

            color = 255 
            x = random.randint(10, 600)
            y = random.randint(10,300)
            w = random.randint(20,30)
            h = random.randint(20,30)
            t = -1

            occluded = truth.copy()
            cv2.rectangle(occluded, (x,y), (x + w, y + h), color, t)

            
            # Check if occluded or not: calculate areas, must be different

            black_pixels_truth = np.sum(truth == 0)
            truth_area = black_pixels_truth/(truth.shape[0]*truth.shape[1])

            black_pixels_occ = np.sum(occluded == 0)
            occluded_area = black_pixels_occ/(occluded.shape[0]*occluded.shape[1])

            while occluded_area >= truth_area-0.001:
                
                x = random.randint(10, 600)
                y = random.randint(10,300)
                w = random.randint(20,30)
                h = random.randint(20,30)

                occluded = truth.copy()
                cv2.rectangle(occluded, (x,y), (x + w, y + h), color, t)

                black_pixels_truth = np.sum(truth == 0)
                truth_area = black_pixels_truth/(truth.shape[0]*truth.shape[1])

                black_pixels_occ = np.sum(occluded == 0)
                occluded_area = black_pixels_occ/(occluded.shape[0]*occluded.shape[1])
            
                #dataset = get_concat_h(truth, occluded)
                #dataset.show()


            # Concatenate 
            newsize = (256, 256)
            dataset = get_concat_h(truth, occluded, newsize)

            #dataset.show()
            name = str(mm) + '_' + str(nn) + '_rect' + '.png'
            dataset.save(path + name)


def circular_occlusions(truth, path= '/home/user/Orientation_learning/Pix2Pix_suole/dataset_augmented_2/'):
        for nn in range(1,30):   # 2 immagini occluse con rettangoli per ogni ground truth

            # Random occlusions generation 

            color = 255 
            r = random.randint(13,25)
            xc = random.randint(50,600)
            yc = random.randint(50,300)
            c = (xc,yc)
            t = -1

            occluded = truth.copy()
            cv2.circle(occluded, c,r,color, t)

            
            # Check if occluded or not: calculate areas, must be different

            black_pixels_truth = np.sum(truth == 0)
            truth_area = black_pixels_truth/(truth.shape[0]*truth.shape[1])

            black_pixels_occ = np.sum(occluded == 0)
            occluded_area = black_pixels_occ/(occluded.shape[0]*occluded.shape[1])

            while occluded_area >= truth_area-0.001:
                
                r = random.randint(15,25)
                xc = random.randint(50,600)
                yc = random.randint(50,300)
                c = (xc,yc)
                t = -1

                occluded = truth.copy()
                cv2.circle(occluded, c,r,color, t)

                black_pixels_truth = np.sum(truth == 0)
                truth_area = black_pixels_truth/(truth.shape[0]*truth.shape[1])

                black_pixels_occ = np.sum(occluded == 0)
                occluded_area = black_pixels_occ/(occluded.shape[0]*occluded.shape[1])
        


            # Concatenate 
            newsize = (256, 256)
            dataset = get_concat_h(truth, occluded, newsize)

            #dataset.show()
            name = str(mm) + '_' + str(nn) + '_circle' + '.png'
            dataset.save(path + name)

def split_train_test(images_dir_in, images_dir_out):
    ''' Create output directories '''
    if not os.path.exists(images_dir_out + 'train/' ):
            os.makedirs(images_dir_out + 'train/' ) 
    if not os.path.exists(images_dir_out + 'test/' ):
            os.makedirs(images_dir_out + 'test/')


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
           shutil.move(images_dir_in + file_name.name, images_dir_out + 'test/'  + file_name.name)


def get_concat_h(im1, im2, newsize):
    im1=Image.fromarray(im1).resize(newsize)
    im2=Image.fromarray(im2).resize(newsize)
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def rotate_contour(cnt, angle):    # Angolo in GRADI
    rect = cv2.minAreaRect(cnt)
    points = cv2.boxPoints(rect)
    points = np.intp(points) 

    cx = rect[0][0]   
    cy = rect[0][1]  

    cnt_norm = cnt - [cx, cy]
    
    coordinates = cnt_norm
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360  #voglio angolo da 0 a 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0] = xs
    cnt_norm[:, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated

def rot_transl_contour(cnt, angle, dx, dy):
    rot_cnt = rotate_contour(cnt, angle)
    rot_tr_cnt = rot_cnt
    rot_tr_cnt[:, 0] = (rot_tr_cnt[:, 0] + dx).astype(np.int32)
    rot_tr_cnt[:, 1] = (rot_tr_cnt[:, 1] + dy).astype(np.int32)

    return rot_tr_cnt

if __name__ == "__main__":

    for jj in range(1,2):     # USO SOLO LA PRIMA IMMAGINE 

        for mm in range (1,101):  # Creo le ground truth: 100 

            name = "/home/user/Orientation_learning/Pix2Pix_suole/masks/" + str(jj) + ".txt"
            mask_data = np.loadtxt(name, dtype=np.int32)


            n = len(mask_data)
            idx_x = np.arange(0,n-1, 2)
            idx_y = np.arange(1,n, 2)
            x = mask_data[idx_x]
            y= mask_data[idx_y]

            contour = np.array([x, y]).T

            # Random rototranslation 

            angle = random.randint(-180,180)
            dx = random.randint(-300,300)
            dy = random.randint(-200,200)

            rot_tr_contour = rot_transl_contour(contour, angle, dx, dy)


            # Check: new ground truth cannot be occluded (move outide se image dimensions)

            conditions = [max(rot_tr_contour[:,0]) >=640, min(rot_tr_contour[:,0]) <= 0, max(rot_tr_contour[:,1]) >= 360, min(rot_tr_contour[:,1]) <= 0]

            while np.any(conditions):
                
                contour = np.array([x, y]).T
                angle = random.randint(-180,180)
                dx = random.randint(-300,300)
                dy = random.randint(-200,200)

                rot_tr_contour = rot_transl_contour(contour, angle, dx, dy)
                conditions = [max(rot_tr_contour[:,0]) >=640, min(rot_tr_contour[:,0]) <= 0, max(rot_tr_contour[:,1]) >= 360, min(rot_tr_contour[:,1]) <= 0]

            truth = np.ones((360,640), dtype=np.uint8 )*255

            cv2.fillPoly(truth, [rot_tr_contour], [0,0,0])
            dataset_path = '/home/user/Orientation_learning/Pix2Pix_suole/dataset_augmented_2/'
            rectangular_occlusions(truth, dataset_path)
            circular_occlusions(truth, dataset_path)



    split_train_test( '/home/user/Orientation_learning/Pix2Pix_suole/dataset_augmented_2/', '/home/user/Orientation_learning/Pix2Pix_suole/dataset_augmented_2/')



    

    










