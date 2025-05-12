#script developed by Damian Dalle Nogare at the Human Technopole Image Analysis Facility
#released under BSD-3 License, 2024

#TODO: fix last quantification part (remove mask reading, but keep intensity image correction)
#TODO: change raw data processing and saving (localz for nuclei, but sum projections for quantifiction channels)


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

cellpose_model = 'models/Elisa_2024_2D'
channels_to_quantify = config['channels']
channel_names = config['channel_names']
dapi_channel = config['dapi_channel']

base_folder = easygui.diropenbox('Select output folder from step 1')
output_folder = easygui.diropenbox('Select folder to store results in')



sum_folder = base_folder + os.path.sep + 'local_z'
os.makedirs(output_folder, exist_ok=True)


dirlist = sorted(os.listdir(base_folder))

while('local_z' in dirlist):
    dirlist.remove('local_z')
while('sum_slices' in dirlist):
    dirlist.remove('sum_slices')


for i, dir in tqdm(enumerate(dirlist)):
    
    restitch_folder = os.path.join(output_folder, dir, 'restitched')
    quantification_folder = os.path.join(output_folder, dir, 'quantification')
    intensity_image_folder = os.path.join(output_folder, dir, 'intensity_images')
    os.makedirs(os.path.join(output_folder, dir), exist_ok=True)
    os.makedirs(restitch_folder, exist_ok=True)
    os.makedirs(intensity_image_folder, exist_ok=True)
    os.makedirs(quantification_folder, exist_ok=True)

    #if not os.path.isdir(output_folder + os.path.sep + dir):
    if os.path.isfile(os.path.join(restitch_folder, 'channel_4.tif')):
        print('file ' + dir + ' already stitched, skipping')
    else:
        print('stitching image ' + dir)
        raw_im = skimage.io.imread(sum_folder + os.path.sep + dir + '.tif')

        os.makedirs(os.path.join(output_folder, dir, 'restitched'), exist_ok=True)
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
                    skimage.io.imsave(os.path.join(output_folder, dir, 'restitched', filename), new_im_crop, check_contrast=False)
                        
    #cellpose segment
    print('segmenting nuclei')
    model = models.CellposeModel(pretrained_model=cellpose_model)
    masks_folder = os.path.join(output_folder, dir, 'cellpose')
    os.makedirs(masks_folder, exist_ok=True)

    if os.path.isfile(masks_folder + os.path.sep + 'masks_' + dir + '.tif'):
        print('file already segmented, skipping')
        masks = skimage.io.imread(os.path.join(masks_folder, 'masks_' + dir + '.tif'))                  
    else:
        print('segmenting ' + dir)
        im = skimage.io.imread(os.path.join(output_folder, dir, 'restitched', 'channel_1.tif'))
        masks, flows, styles  = model.eval(im, diameter=None, flow_threshold=None, channels=[0,0])
    
        organoid_mask = skimage.io.imread( os.path.join(base_folder, dir, 'slice_mask.tif'))
         
        binary = masks + organoid_mask
        binary[binary > 0] = 1

        h, w = binary.shape
        border_mask = np.ones_like(binary, dtype=bool)
        border_mask[int(h*0.03):int(h*0.97), int(w*0.03):int(w*0.97)] = False       #within 3% of border
        labs = skimage.measure.label(binary)
        labs = labs.astype(np.uint16)
        border_labs = np.unique(labs[border_mask])
        for border in border_labs:
            labs[labs == border] = 0

        labs[labs > 0] = 1

        mask_im = np.multiply(masks, labs)


        skimage.io.imsave(os.path.join(masks_folder, 'masks_' + dir + '.tif'), mask_im, check_contrast=False)                                   



    files = os.listdir(restitch_folder)

    
    #df = pd.DataFrame()


    for position, channel in enumerate(channels_to_quantify):
        measure_im = skimage.io.imread(os.path.join(output_folder, dir, 'restitched', 'channel_' + str(channel) + '.tif'))
       
        stats = skimage.measure.regionprops_table(mask_im, intensity_image=measure_im, properties=['label', 'mean_intensity'])
        if not os.path.isfile(os.path.join(intensity_image_folder, 'channel_' + str(channel) + '.tif')):
            label_to_mean_intensity = {label: mean_intensity for label, mean_intensity in zip(stats['label'], stats['mean_intensity'])}
            label_to_mean_intensity[0] = 0
            intensity_im = np.vectorize(label_to_mean_intensity.get)(mask_im)

            skimage.io.imsave(os.path.join(intensity_image_folder, 'channel_'  + str(channel) + '.tif'), intensity_im.astype(np.uint16), check_contrast=False)
        rounded_intensity = [ '%.2f' % elem for elem in stats['mean_intensity'] ]

print('pipeline finished')

