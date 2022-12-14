import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
    """ Load the test images """
    test_images = glob("test_images/*")

    """ Load the model """
    model = tf.keras.models.load_model("unet.h5")
    print(model.summary())

    for path in tqdm(test_images, total=len(test_images)):
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        original_image = x
        h, w, _ = x.shape

        x = cv2.resize(x, (256, 256))
        x = x/255.0
        x = x.astype(np.float32)

        x = np.expand_dims(x, axis=0)
        pred_mask = model.predict(x)[0]

        pred_mask = np.concatenate(
            [
                pred_mask,
                pred_mask,
                pred_mask
            ], axis=2)
        pred_mask = (pred_mask > 0.25) * 255
        pred_mask = pred_mask.astype(np.float32)
        pred_mask = cv2.resize(pred_mask, (w, h))
        
        pred_mask[np.where((pred_mask==[255, 255, 255]).all(axis=2))] = [0, 0, 255]
        #cv2.imshow('output',pred_mask)
        #cv2.waitKey(0)


        original_image = original_image.astype(np.float32)

        alpha = 0.5

        cv2.addWeighted(pred_mask, alpha, original_image, 1-alpha, 0, original_image)
        name = path.split("/")[-1]
        name=name.split('.jpg')[0]+'_masked.jpg'
        cv2.imwrite(f"{name}", original_image)