import cv2
import os
import numpy as np
rgb_dir="/home/xu/data/png4"
mask_dir="/home/xu/data/mask4"
out_dir_dense="/home/xu/data/view_synthesis/object_dense"
out_dir="/home/xu/data/view_synthesis/object"

t = 128
count = 0
for root, folders, files in os.walk(rgb_dir):
    for file in sorted(files):
        if 'r0.png' in file:
            model_id = int(file.split('_')[0])
            if ( model_id > 700 ):
                continue
            for view in np.arange(0,355,5):

                abs_path = os.path.join(root,file)
                abs_path = abs_path.replace('_r0.png', '_r%d.png'%view)
                img = cv2.imread(abs_path)

                mask = cv2.imread(abs_path.replace(rgb_dir, mask_dir))
                if view == 0:
                    mask = cv2.imread(abs_path.replace(rgb_dir, mask_dir).replace("r0.png","c1.png"))
                img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img[img_gray<10] = 128
                h,w,c = img.shape
                img = img[ (h-t)/2:(h+t)/2, (w-t)/2:(w+t)/2, :]

                # img = np.pad(img,((0,0),(24,24),(0,0)), 'constant',constant_values=128)

                # img = np.pad(img,((0,0),(24,24),(0,0)), 'constant',constant_values=128)

                out_folder_dense = os.path.join(out_dir_dense, str(view) )

                if np.mod(model_id,70):
                    if not os.path.exists(out_folder_dense):
                        os.makedirs(out_folder_dense)
                    out_path_dense = os.path.join(out_folder_dense, file)
                    cv2.imwrite(out_path_dense, img)

                elif np.mod(view,20) == 0:
                    out_folder = os.path.join(out_dir, str(view/20))

                    if not os.path.exists(out_folder):
                        os.makedirs(out_folder)
                    output_path= os.path.join(out_folder, file)
                    cv2.imwrite(output_path, img)


