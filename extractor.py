import os
import numpy as np 
import cv2
from glob import glob
from tqdm import tqdm
import json
import pandas as pd 

def get_all_images_and_jsons_path(path_img):
    data_locations=pd.DataFrame(columns=['images'])
    data_locations['images']=glob(path_img+'/*.jpg')
    return data_locations

def get_all_cords(axis,file_path,images):
    cords=[]
    if axis=='x':
        look_for="all_points_x"
    else: 
        look_for="all_points_y"
    
    file_path=glob(file_path+'/*.json')
    file_path=file_path[0]
    with open(file_path,'rb') as f:
        f=json.load(f)
    json_file=f['_via_img_metadata']
    images=[os.path.basename(i) for i in images]
    
    new_json_={}
    for key in json_file.keys():
        key_changed=key.split('jpg')[0]
        new_json_[key_changed] = json_file[key]
    
    json_file=new_json_
    for index in images:
        index=index.split('jpg')[0]
        if index in json_file.keys(): 
            json_file_list=json_file[index]['regions']
            t_cords=[]
            for regions in json_file_list: 
                t_cords.append(regions['shape_attributes'][look_for])
            cords.append(t_cords)
    return cords
            


def get_all_images_and_cords(input_dataframe,path_jsons):
    data_and_cords=pd.DataFrame(columns=['images','x','y'])
    data_and_cords['images']=input_dataframe['images']
    data_and_cords['x']=get_all_cords('x',path_jsons,data_and_cords['images'].tolist())
    data_and_cords['y']=get_all_cords('y',path_jsons,data_and_cords['images'].tolist())
    return data_and_cords



def get_all_cords_in_x_y(x_list,y_list):
    cords=[]
    for i in range(len(x_list)):
        cords.append([x_list[i],y_list[i]])
    cords=np.array(cords,np.int32)

    return cords

def get_masked_images(df):
    try: 
        os.mkdir('masked_images')
    except:
        pass 
    try: 
        images=df['images'].tolist()
        cord_x=df['x'].tolist()
        cord_y=df['y'].tolist()
        for i in range(len(images)):
            name=os.path.basename(images[i])
            image=cv2.imread(images[i])
            h,w,c=image.shape
            blank_image = np.zeros((h,w,c), np.uint8)
            cv2.imwrite('temp.jpg',blank_image)
            img=cv2.imread('temp.jpg')
            for j in range(len(cord_x[i])):
                cords=get_all_cords_in_x_y((cord_x[i])[j],(cord_y[i])[j])        
                img=cv2.fillPoly(img, pts =[cords], color=(255,255,255))
            cv2.imwrite('masked_images\\'+name,img)        
            print(str(i+1)+' out of '+str(len(images))+' Completed !')
            os.remove('temp.jpg')
    except:
        print('Error occured ')
        if 'temp.jpg' in os.listdir(os.getcwd()):
            os.remove('temp.jpg')  

if __name__=='__main__':
    '''
        Initially we take input paths for images and 
        annotations obtained from vga annotator tool 
        Note: Annotations should have same number of annotation as images present in the path of image folder.
    '''
    path_images=input('Enter path for all images: ')
    path_jsons=input('Enter path for all jsons: ')
    '''
        We are using going to get all the images on the image folder.
        get_all_images_and_jsons_path(path for images)
    '''
    table_with_images_path=get_all_images_and_jsons_path(path_images)
    #print("CHECKPOINT 1")
        
    '''
        Get all the images path and also their coresponding coords for mask obtained from annotations
        get_all_images_and_cords(table with images, json file path)
    '''
    table_with_images_and_cords=get_all_images_and_cords(table_with_images_path,path_jsons) 
    #print("CHECKPOINT 2")
    '''
        Take all the image paths and their corresponding x and y axis for masks 
        Create mask for all the image with annotations.
    '''
    get_masked_images(table_with_images_and_cords)