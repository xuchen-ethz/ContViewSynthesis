import cv2
import os
import numpy as np
rgb_dir="/home/xu/data/png4"
mask_dir="/home/xu/data/mask4"
out_dir="/home/xu/data/object"
t = 255

for root, folders, files in os.walk(rgb_dir):
    for file in sorted(files):
        if 'r0.png' in file:
            for view in np.arange(0,355,5):

                abs_path = os.path.join(root,file)
                abs_path = abs_path.replace('_r0.png', '_r%d.png'%view)
                img = cv2.imread(abs_path)

                mask = cv2.imread(abs_path.replace(rgb_dir, mask_dir))
                if view == 0:
                    mask = cv2.imread(abs_path.replace(rgb_dir, mask_dir).replace("r0.png","c1.png"))

                img[mask==0] = 128
                h,w,c = img.shape

                img = np.pad(img,((0,0),(24,24),(0,0)), 'constant',constant_values=128)

                out_folder = os.path.join(out_dir, str(view) )

                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)
                out_path = os.path.join(out_folder, file)
                cv2.imwrite(out_path, img)
                # cv2.imshow("s",img)
                # cv2.waitKey()


