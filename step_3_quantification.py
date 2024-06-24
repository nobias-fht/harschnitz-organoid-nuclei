from tqdm import tqdm
import skimage
import numpy as np
import yaml
import os
import pandas as pd
import easygui

def calculate_all_double_positives(ch1, ch2, df):
    double_pos = 0
    for i in range(0, len(df['label'])):
        if df.loc[i, 'positives_ch' + str(ch1)] == 1:
            if df.loc[i, 'positives_ch' + str(ch2)] == 1:
                double_pos = double_pos + 1
    return double_pos


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)




#output_folder = 'step_2_output_new'
output_folder = easygui.diropenbox('Select folder output from step 2')
image_folder = output_folder + os.path.sep + 'restitched'
positive_image_folder = output_folder + os.path.sep + 'positive_cell_images'
quantification_folder = output_folder + os.path.sep + 'quantification'
intensity_image_folder = output_folder + os.path.sep + 'intensity_images'
masks_folder = output_folder + os.path.sep + 'cellpose'
report_folder = output_folder + os.path.sep + 'report'
os.makedirs(report_folder, exist_ok=True)
os.makedirs(positive_image_folder, exist_ok=True)
files = os.listdir(image_folder)
for i, file in enumerate(files):
    CONFIG_NAME = 'config.yaml'
    with open(CONFIG_NAME, "r") as f:
        config = yaml.safe_load(f)

    channels_to_quantify = config['channels']
    weighting_d25 = config['weighting_d25']
    weighting_d50 = config['weighting_d50']
    weighting_d75 = config['weighting_d75']
    channel_names = config['channel_names']
    channel_names.insert(0, 'total')
    print(channel_names)
    df = pd.read_csv(quantification_folder + os.path.sep + file + '.csv')
    df_thresh = pd.read_csv(quantification_folder + os.path.sep + file + '_thresh.csv')
    thresholds = []
    positives = []
    total_positive_list = []
    total_cells = len(df['label'])
    total_positive_list.append(total_cells)
    percentages = []
    percentages.append(1)
    thresholds.append(-1)
    print(file)
    df_summary = pd.DataFrame()

    for position, channel in enumerate(channels_to_quantify):
        positive_list = []
        positive_count = 0
        if 'Day_25' in file:
            weighting = weighting_d25
        if 'Day_50' in file:
            weighting = weighting_d50
        if 'Day_75' in file:
            weighting = weighting_d75

        df_positives = pd.DataFrame()
        thresh = df_thresh['bg_mean_ch_' + str(channel)] + (weighting[position] * df_thresh['bg_sd_ch_' + str(channel)])
        thresh = thresh[0]
        print('using threshold: ' + str(thresh))
        thresholds.append(int(thresh))
        

        for intensity in df['intensity_ch_' + str(channel)]:
            if intensity <= thresh:
                positive_list.append(0)
            if intensity > thresh:
                positive_list.append(1)
                positive_count = positive_count + 1
        df['positives_ch' + str(channel)] = positive_list
        total_positive_list.append(positive_count)
        percentages.append(round((positive_count / total_cells), 2))
    
        #make the channel images
        intensity_image = skimage.io.imread (intensity_image_folder + os.path.sep + file + '_ch' + str(channel) + '.tif')
        intensity_image[intensity_image<thresh] = 0
        intensity_image[intensity_image>thresh] = channel-1
        skimage.io.imsave(positive_image_folder + os.path.sep + file + '_positive_ch' + str(channel) + '.tif', intensity_image, check_contrast=False)

    #calculate all double positives
    
    channel_names.append(channel_names[0+1] + '+' + channel_names[1+1])
    double_pos = calculate_all_double_positives(channels_to_quantify[0], channels_to_quantify[1], df)
    total_positive_list.append(double_pos)
    percentages.append(round((double_pos / total_cells), 2))
    thresholds.append(-1)

    channel_names.append(channel_names[0+1] + '+' + channel_names[2+1])
    double_pos = calculate_all_double_positives(channels_to_quantify[0], channels_to_quantify[2], df)
    total_positive_list.append(double_pos)
    percentages.append(round((double_pos / total_cells), 2))
    thresholds.append(-1)
    
    channel_names.append(channel_names[1+1] + '+' + channel_names[2+1])
    double_pos = calculate_all_double_positives(channels_to_quantify[1], channels_to_quantify[2], df)
    total_positive_list.append(double_pos)
    percentages.append(round((double_pos / total_cells), 2))
    thresholds.append(-1)
    print(channel_names)
    df_summary['channels'] = channel_names
    df_summary['threshold'] = thresholds
    df_summary['count'] = total_positive_list
    df_summary['percent'] = percentages
    
    df_summary.to_csv(report_folder + os.path.sep + file + '_summary.csv', index=False)