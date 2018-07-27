import cv2
import os
import numpy as np
in_dir="/home/xu/Downloads/epfl-gims08/tripod-seq"
out_dir="/home/xu/data/car"
prefix = "tripod_seq_"

t = 256
for seq_id in np.arange(1,20):
    bboxes = np.loadtxt( os.path.join(in_dir,'bbox_%02d.txt'%seq_id)).astype(int)
    for fid in np.arange(1, bboxes.shape[0]):
        img_path = os.path.join(in_dir, prefix+"%02d_%03d.jpg"%(seq_id,fid))
        img = cv2.imread(img_path)
        x,y,w,h = bboxes[fid-1,:]
        if w < t: x = x- (t-w)/2; w = t;
        if h < t: y = y- (t-h)/2; h = t;
        print x,y,w,h
        img = img[y:y+h,x:x+w,:]

        cv2.imshow('img',img)
        cv2.waitKey()
#
# for root, folders, files in os.walk(rgbd_dataset):
#     for file in sorted(files):
#         if '001.jpg' in file:
#             img_views = []
#             skip = False
#             for view in np.arange(1,80,4):
#
#                 abs_path = os.path.join(root,file)
#                 abs_path = abs_path.replace('_1_crop.png', '_%d_crop.png'%view)
#                 img = cv2.imread(abs_path)
#                 if img is None:
#                     skip = True
#                     break
#                 mask = cv2.imread(abs_path.replace("_crop.png", "_maskcrop.png"))
#                 img[mask==0] = 128
#                 h,w,c = img.shape
#
#                 ratio = t / float(np.max( np.array([h,w,t])))
#
#                 h_new,w_new = int(h*ratio), int(w*ratio)
#                 img = cv2.resize(img, (h_new, w_new))
#                 h,w,c = img.shape
#
#                 img = np.pad(img,(((t-h)/2,(t-h)/2),((t-w)/2,(t-w)/2),(0,0)), 'constant',constant_values=128)
#                 img_views.append(img)
#
#             if skip: continue
#
#             for n,img in enumerate(img_views):
#                 out_folder = os.path.join(out_dir, str(n) )
#                 if not os.path.exists(out_folder):
#                     os.makedirs(out_folder)
#                 out_path = os.path.join(out_folder, file)
#                 cv2.imwrite(out_path, img)
