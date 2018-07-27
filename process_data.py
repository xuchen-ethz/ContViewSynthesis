import cv2
import os
import numpy as np
rgbd_dataset="/home/xu/data/rgbd-dataset"
out_dir="/home/xu/data/rbgd_processed"
t = 128
for root, folders, files in os.walk(rgbd_dataset):
    for file in sorted(files):
        if '_1_crop.png' in file:
            img_views = []
            skip = False
            for view in np.arange(1,180,10):

                abs_path = os.path.join(root,file)
                abs_path = abs_path.replace('_1_crop.png', '_%d_crop.png'%view)
                img = cv2.imread(abs_path)
                if img is None:
                    skip = True
                    break
                mask = cv2.imread(abs_path.replace("_crop.png", "_maskcrop.png"))
                img[mask==0] = 128
                h,w,c = img.shape

                ratio = t / float(np.max( np.array([h,w,t])))

                h_new,w_new = int(h*ratio), int(w*ratio)
                img = cv2.resize(img, (h_new, w_new))
                h,w,c = img.shape

                img = np.pad(img,(((t-h)/2,(t-h)/2),((t-w)/2,(t-w)/2),(0,0)), 'constant',constant_values=128)
                img_views.append(img)

            if skip: continue

            for n,img in enumerate(img_views):
                out_folder = os.path.join(out_dir, str(n) )
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)
                out_path = os.path.join(out_folder, file)


                cv2.imwrite(out_path, img)
                # cv2.imshow('s',img)
                # cv2.waitKey(0)

