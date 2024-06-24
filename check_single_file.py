#script developed by Damian Dalle Nogare at the Human Technopole Image Analysis Facility
#released under BSD-3 License, 2024

print('starting pipeline')
print('importing libraries')
import skimage
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import easygui
import napari
import yaml

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

CONFIG_NAME = 'config.yaml'
with open(CONFIG_NAME, "r") as f:
	config = yaml.safe_load(f)
	
channels = config['channels']
weighting_d25 = config['weighting_d25']
weighting_d50 = config['weighting_d50']
weighting_d75 = config['weighting_d75']
channel_names = config['channel_names']

#get user inputs
file_to_process = easygui.diropenbox('Select file to process')
os.chdir(file_to_process)
filename_split = file_to_process.split('/')
file = filename_split[len(filename_split)-1]

if 'Day_25' in file:
    weighting = weighting_d25
if 'Day_50' in file:
    weighting = weighting_d50
if 'Day_75' in file:
    weighting = weighting_d75

print('weighting: ' + str(weighting))

print('loading viewer')
viewer = napari.Viewer()


head, tail = os.path.split(file_to_process)
head, tail = os.path.split(head)
mask_path = head + os.path.sep + 'cellpose'
mask = skimage.io.imread(mask_path + os.path.sep + 'masks_' + file + '.tif')

for i, channel in enumerate(channels):
    measure_im = skimage.io.imread(file_to_process + os.path.sep + 'channel_' + str(channel) + '.tif')
    print('loaded file ' + file_to_process)
    thresh_df = pd.read_csv(head + os.path.sep + 'quantification' + os.path.sep + file + '_thresh.csv')
    intensity_image_folder = os.path.join(head, 'intensity_images')
    thresh = thresh_df['bg_mean_ch_' + str(channel)] + (weighting[i] * thresh_df['bg_sd_ch_' + str(channel)])   #((0.1 * thresh_df['slice_sd_ch_' + str(chan)][0]))
    os.makedirs(intensity_image_folder, exist_ok=True)
    stats = skimage.measure.regionprops_table(mask, intensity_image=measure_im, properties=['label', 'mean_intensity'])

    if not os.path.isfile(intensity_image_folder + os.path.sep + file + '_ch' + str(channel) + '.tif'):
        intensity_image = np.zeros(mask.shape, dtype=np.uint16)
        for i, lab in enumerate(tqdm(stats['label'])):
            intensity_image[mask == lab] = stats['mean_intensity'][i]
        skimage.io.imsave(intensity_image_folder + os.path.sep + file  + '_ch' + str(channel) + '.tif', intensity_image, check_contrast=False)

    else:
        intensity_image = skimage.io.imread(intensity_image_folder + os.path.sep + file + '_ch' + str(channel) + '.tif')

    intensity_image[intensity_image<thresh[0]] = 0
    intensity_image[intensity_image>thresh[0]] = channel-1


    if i == 0:
        viewer.add_labels(intensity_image.astype(np.uint16), blending='additive', name="positive_ch" + str(channel))
        viewer.add_image(measure_im.astype(np.uint16), blending='additive', name="intensity_ch" + str(channel))
        viewer.layers['intensity_ch' + str(channel)].contrast_limits = (0, 65500)
    else: 
        viewer.add_labels(intensity_image.astype(np.uint16), blending='additive', name="positive_ch" + str(channel), visible=False)
        viewer.add_image(measure_im.astype(np.uint16), blending='additive', name="intensity_ch" + str(channel), visible=False)
        viewer.layers['intensity_ch' + str(channel)].contrast_limits = (0, 65500)
input("Press Enter to continue...")

print('script finished')
