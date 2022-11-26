import imgaug.augmenters as ima
import cv2
import glob 

flip_h=ima.Sequential([
    ima.Fliplr(1)
])

flip_v=ima.Sequential([
    ima.Flipud(1)
])


rotate=ima.Sequential([
    ima.Rotate(-70)
])


rotate_r=ima.Sequential([
    ima.Rotate(70)
])


affine=ima.Sequential([
    ima.Affine(translate_percent={'x':0.2,'y':0.2},rotate=-30)
])

affine_2=ima.Sequential([
    ima.Affine(translate_percent={'x':-0.2,'y':-0.2},rotate=30)
])

affine_3=ima.Sequential([
    ima.Affine(translate_percent={'x':0.4,'y':-0.2},rotate=30)
])


scale=ima.Sequential(
    ima.Resize(1.5)
)

scale_d=ima.Sequential(
    ima.Resize(0.5)
)

def augment_all_(path_files):
    all_images=glob.glob(path_files+'\*.jpg')
    num=0
    for image_name in all_images:
        print(str(num)+' out of '+str(len(all_images)))
        image_og=cv2.imread(image_name)
        
        image=flip_h(image=image_og)
        #print(image_name.split('.jpg')[0]+'_fliped.jpg')
        cv2.imwrite(image_name.split('.jpg')[0]+'_fliped_h.jpg',image)

        image=flip_v(image=image_og)
        cv2.imwrite(image_name.split('.jpg')[0]+'_fliped_v.jpg',image)

        image=rotate(image=image_og)
        cv2.imwrite(image_name.split('.jpg')[0]+'_rotated_l.jpg',image)

        image=rotate_r(image=image_og)
        cv2.imwrite(image_name.split('.jpg')[0]+'_rotated_r.jpg',image)

        image=affine(image=image_og)
        cv2.imwrite(image_name.split('.jpg')[0]+'_afined.jpg',image)

        image=affine_2(image=image_og)
        cv2.imwrite(image_name.split('.jpg')[0]+'_afined_2.jpg',image)

        image=affine_3(image=image_og)
        cv2.imwrite(image_name.split('.jpg')[0]+'_afined_3.jpg',image)

        image=scale(image=image_og)
        cv2.imwrite(image_name.split('.jpg')[0]+'_scaled_up.jpg',image)
        
        image=scale_d(image=image_og)
        cv2.imwrite(image_name.split('.jpg')[0]+'_scaled_down.jpg',image)
        num+=1

if __name__=='__main__':
    images_path=[]
    images_path.append(input('Enter path for images: '))
    images_path.append(input('Enter path for masks: '))
    print('ALL IMAGES: \n')
    augment_all_(images_path[0])
    print('ALL MASKS: \n')
    augment_all_(images_path[1])