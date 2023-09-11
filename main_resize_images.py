import glob
import json
import numpy as np
import os
import skimage.io
import skimage.transform
import time
import tqdm

main_folder = 'C:/Vejdirektorat/dataset_exploration/images_for_annotation' #'D:/VJD_data/raw'
output_main_folder = 'C:/Vejdirektorat/dataset_exploration'

# main_folder = '/home/anders/ST_AU234627/Research/2020_Vejdirektorat_invasive_plantearter/images_for_annotation'
# output_main_folder = '/home/anders/ST_AU234627/Research/2020_Vejdirektorat_invasive_plantearter/'

new_image_size = (96,128) #VJD5: (96,128) #VJD4: (384,512) #VJD3: (768,1024) #VJD2: (1536, 2048) #VJD: (192,256)

overwrite_if_exist = False


output_resize_folder = os.path.join(output_main_folder, 'resize_' + 'x'.join([str(s).zfill(4) for s in new_image_size]))

subfolder_paths = glob.glob(os.path.join(main_folder,'*'))

img_load_times = []
img_resize_times = []
img_save_times = []
for subfolder_path in tqdm.tqdm(subfolder_paths):
    _, subfolder_name = os.path.split(subfolder_path)
    subfolder_out = os.path.join(output_resize_folder, subfolder_name)
    os.makedirs(subfolder_out, exist_ok=True)

    image_paths = glob.glob(os.path.join(subfolder_path, '*.jpg'))
    for image_path in tqdm.tqdm(image_paths):
        
        _, image_name = os.path.split(image_path)
        image_path_out = os.path.join(subfolder_out, image_name)

        if (overwrite_if_exist) | (not os.path.isfile(image_path_out)):
        # Wrap in try to handle corrupt files
            try:
                t0 = time.time()
                I = skimage.io.imread(image_path)
                t1 = time.time()
                Ir = skimage.transform.resize(I, new_image_size, preserve_range=True, anti_aliasing=True)
                t2 = time.time()
                skimage.io.imsave(image_path_out, Ir.astype(np.uint8), quality=100)
                t3 = time.time()
            except Exception as e:
                print('\nAn exception ('+ str(e.__class__) +') occured while processing image (' + image_name + '):')
                print(e)
                pass
            else:
                # Only append if no error occured
                img_load_times.append(t1-t0)
                img_resize_times.append(t2-t1)
                img_save_times.append(t3-t2)

##

timing_dict = {'time_load': img_load_times, 'time_resize': img_resize_times, 'time_save': img_save_times, 'image_resize': new_image_size}
fob = open(os.path.join(output_resize_folder, 'timing_dict.json'),'w')
json.dump(timing_dict, fob, indent=4)
fob.close()

print('Load time:', np.min(img_load_times), np.mean(img_load_times), np.max(img_load_times))
print('Resize time:', np.min(img_resize_times), np.mean(img_resize_times), np.max(img_resize_times))
print('Save time:', np.min(img_save_times), np.mean(img_save_times), np.max(img_save_times))