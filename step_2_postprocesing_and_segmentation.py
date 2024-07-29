#script developed by Damian Dalle Nogare at the Human Technopole Image Analysis Facility
#released under BSD-3 License, 2024

print('starting pipeline')
print('importing libraries')
import skimage
import numpy as np
import yaml
import os
from tqdm import tqdm
from cellpose import models
import pandas as pd
import easygui
print('imports finished')

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print(os.getcwd())

CONFIG_NAME = 'config.yaml'
with open(CONFIG_NAME, "r") as f:
	config = yaml.safe_load(f)

cellpose_model = config['cellpose_model']
channels_to_quantify = config['channels']
channel_names = config['channel_names']

base_folder = easygui.diropenbox('Select output folder from step 1')
output_folder = easygui.diropenbox('Select folder to store results in')
sum_folder = base_folder + os.path.sep + 'local_z'
quantification_folder = output_folder + os.path.sep + 'quantification'
intensity_image_folder = os.path.join(output_folder + os.path.sep + 'intensity_images')
os.makedirs(output_folder, exist_ok=True)
os.makedirs(quantification_folder, exist_ok=True)
os.makedirs(intensity_image_folder, exist_ok=True)

dirlist = sorted(os.listdir(base_folder))

while('local_z' in dirlist):
    dirlist.remove('local_z')
while('sum_slices' in dirlist):
    dirlist.remove('sum_slices')

print('restitching images')
for i, dir in tqdm(enumerate(dirlist)):
    #if not os.path.isdir(output_folder + os.path.sep + dir):
    if os.path.isfile(output_folder + os.path.sep + 'restitched' + os.path.sep + dir + os.path.sep + 'channel_4.tif'):
        print('file already stitched, skipping')
        continue
    else:
        print('stitching image ' + dir)
        raw_im = skimage.io.imread(sum_folder + os.path.sep + dir + '.tif')

        os.makedirs(output_folder + os.path.sep + 'restitched' + os.path.sep + dir, exist_ok=True)
        CONFIG_NAME = base_folder + os.path.sep + dir + os.path.sep + 'data.yml'
        with open(CONFIG_NAME, "r") as f:
            config = yaml.safe_load(f)

        filelist = sorted(os.listdir(base_folder + os.path.sep + dir))
        for file in filelist:
            if file[-4:] == '.tif':
                if file != 'slice_mask.tif':
                    im = skimage.io.imread(base_folder + os.path.sep + dir + os.path.sep + file)
                    Xtiles = config['XTiles']
                    Ytiles = config['YTiles']
                    XOverlap = int(config['XOverlap']) 
                    YOverlap = int(config['YOverlap']) 
                    w = im.shape[2]
                    h = im.shape[1]
                    full_w = w * Xtiles  + 5000
                    full_h = h * Ytiles  + 5000
                    new_im = np.zeros((full_w, full_h), dtype=np.uint16)
                    #print(im.shape)
                    
                    #check to see which the smallest axis is and reshape if necessary
                    if im.shape[0] < im.shape[2]:
                        imshape = im.shape[0]

                    else:
                        imshape = im.shape[2]
                        im = np.moveaxis(im, 2, 0)
                        #print(im.shape) 

                    for j in range(imshape):
                        xpos = config['tile_' + str(j) + '_xpos']
                        ypos = config['tile_' + str(j) + '_ypos']
                        xend = config['tile_' + str(j) + '_xpos'] +  im.shape[2]
                        yend =  config['tile_' + str(j) + '_ypos'] +  im.shape[1]                      
                        new_im[int(ypos):int(yend), int(xpos):int(xend)] = im[j,:,:]
                    new_im_crop = np.copy(new_im[0:raw_im.shape[0], 0:raw_im.shape[1]])
                    filename = 'channel_' + file[-5] + '.tif'
                    skimage.io.imsave(output_folder + os.path.sep + 'restitched' + os.path.sep + dir + os.path.sep + filename, new_im_crop, check_contrast=False)
                        
#cellpose segment
print('segmenting nuclei')
model = models.CellposeModel(pretrained_model=cellpose_model)
masks_folder = output_folder + os.path.sep + 'cellpose'
os.makedirs(masks_folder, exist_ok=True)
dirlist = os.listdir(output_folder + os.path.sep + 'restitched')

for dir in dirlist:
    if os.path.isfile(masks_folder + os.path.sep + 'masks_' + dir + '.tif'):
        print('file already segmented, skipping')
    else:
        print('segmenting ' + dir)
        im = skimage.io.imread(output_folder + os.path.sep + 'restitched' + os.path.sep + dir + os.path.sep + 'channel_1.tif')
        masks, flows, styles  = model.eval(im, diameter=None, flow_threshold=None, channels=[0,0])
        skimage.io.imsave(masks_folder + os.path.sep + 'masks_' + dir + '.tif', masks, check_contrast=False)                                   

image_folder = output_folder + os.path.sep + 'restitched'
files = os.listdir(image_folder)

for i, file in enumerate(files):
        print('quantifying ' + file)

        mask = skimage.io.imread(masks_folder + os.path.sep + 'masks_' + file + '.tif')
        organoid_mask = skimage.io.imread(base_folder + os.path.sep + file + os.path.sep + 'slice_mask.tif')

        if mask.shape[0] != organoid_mask.shape[0] or mask.shape[1] != organoid_mask.shape[1]:
            organoid_mask = skimage.transform.resize(organoid_mask,
                        mask.shape,
                        mode='edge',
                        anti_aliasing=False,
                        anti_aliasing_sigma=None,
                        preserve_range=True,
                        order=0)

        thresh_df = pd.DataFrame()
        df = pd.DataFrame()

        for position, channel in enumerate(channels_to_quantify):
            marked = np.zeros(mask.shape, dtype=np.uint16)
            measure_im = skimage.io.imread(image_folder + os.path.sep + file + os.path.sep + 'channel_' + str(channel) + '.tif')
            pixels_in_mask = measure_im[organoid_mask > 0]
            background_array = measure_im[organoid_mask == 0]
            nonzero_backround = background_array[background_array > 0]
            background_mean = (np.mean(nonzero_backround))
            background_sd = (np.std(nonzero_backround))
            thresh_df['bg_mean_ch_' + str(channel)] = [background_mean]
            thresh_df['bg_sd_ch_' + str(channel)] = [background_sd]
            stats = skimage.measure.regionprops_table(mask, intensity_image=measure_im, properties=['label', 'mean_intensity'])
            if not os.path.isfile(intensity_image_folder + os.path.sep + file + '_ch' + str(channel) + '.tif'):
                intensity_image = np.zeros(mask.shape, dtype=np.uint16)
                for i, lab in enumerate(tqdm(stats['label'])):
                    intensity_image[mask == lab] = int(stats['mean_intensity'][i])
                skimage.io.imsave(intensity_image_folder + os.path.sep + file  + '_ch' + str(channel) + '.tif', intensity_image, check_contrast=False)
            rounded_intensity = [ '%.2f' % elem for elem in stats['mean_intensity'] ]
            df['label'] = stats['label']
            df['intensity_ch_' + str(channel)] = rounded_intensity
            df.to_csv(quantification_folder + os.path.sep + file + '.csv')
            thresh_df.to_csv(quantification_folder + os.path.sep + file + '_thresh.csv')
print('pipeline finished')

