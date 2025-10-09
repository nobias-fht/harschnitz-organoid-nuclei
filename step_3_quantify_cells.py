import napari
from qtpy.QtWidgets import QPushButton, QSlider, QVBoxLayout, QWidget, QLineEdit, QLabel, QComboBox
from qtpy.QtCore import Qt
import easygui
import numpy as np
from skimage.io import imread
import os
import skimage
from biapy import BiaPy
import pandas as pd
import yaml
import shutil


size_threshold = 100

global last_path

global global_multiplier
global seg_method
global_multiplier = 0
df = pd.DataFrame()

CONFIG_NAME = 'config.yaml'
with open(CONFIG_NAME, "r") as f:
	config = yaml.safe_load(f)

last_path = '/facility/imganfac/neurogenomics/harschnitz/Elisa_Colombo/background_removal/pre-deployment-test/step2/'


#last_path = os.getcwd()

 
dapi_channel = config['dapi_channel']
seg_method = 'otsu'

available_models = os.listdir('checkpoints')
available_models = [x for x in available_models if x.endswith('.pth')]



def on_unet_button_click():
    global last_path
    print('run u-net')

    #delete the current layer 'unet segmentation'
    layer = next((layer for layer in viewer.layers if layer.name == "unet_segmentation"), None)
    if layer:
        viewer.layers.remove(layer)
    #delete the current layer 'unet filtered cellpose results'
    layer = next((layer for layer in viewer.layers if layer.name == "unet filtered cellpose results"), None)
    if layer:
        viewer.layers.remove(layer)

    model_path = 'checkpoints/unet.pth'

    #model_path = easygui.fileopenbox(title="Select Model Checkpoint File")
    yaml_file = model_path[:-4] + '.yaml'
    print('model path: ' + str(model_path))
    print('yaml path: ' + str(yaml_file))
   
    input_path = os.path.join(last_path, 'unet_temp')
    output_path = os.path.join(last_path, 'unet_output')
    fake_gt_path = os.path.join(output_path, 'fake_gt')

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(fake_gt_path, exist_ok=True)
    
    head, tail = os.path.split(model_path)
    
    output_file = os.path.join(output_path, tail[:-4], 'results', tail[:-4] + '_1', 'per_image_binarized', 'raw_temp.tif')
    dst_file =  os.path.join(output_path, 'unet_seg.tif')

    #copy output file to output path
    #if not os.path.exists(os.path.join(output_path, tail[:-4])):
    #    shutil.copy(output_file, dst_file)


    threshold = float(text_box_unet_threshold.text())

    if not os.path.isfile(dst_file):
        intensity_layer = next((layer for layer in viewer.layers if 'channel' in layer.name), None).data
        skimage.io.imsave(os.path.join(last_path, 'unet_temp', 'raw_temp.tif'), intensity_layer, check_contrast=False)
        yaml_file, job_name = make_config(input_path, yaml_file, model_path, output_path, fake_gt_path)
        biapy = BiaPy(yaml_file, result_dir=output_path, name=job_name, run_id=1, gpu=0)
        biapy.run_job()
        shutil.copy(output_file, dst_file)
    else:
        print('u-net segmented file already exists, skipping')

    unet_seg = skimage.io.imread(dst_file)
    unet_seg[unet_seg == 2] = 0
    viewer.add_labels(unet_seg, name='unet_segmentation', blending='additive', visible=False)

    seg = np.copy(viewer.layers['segmentation'].data)
    thresholded = viewer.layers['thresholded'].data

    props = skimage.measure.regionprops_table(seg, intensity_image=unet_seg, properties=['label', 'mean_intensity'])
    label_to_mean_intensity = {label: mean_intensity for label, mean_intensity in zip(props['label'], props['mean_intensity'])}
    labels_to_keep = {label for label, mean_intensity in label_to_mean_intensity.items() if mean_intensity > threshold}
    filtered_label_image = np.where(np.isin(seg, list(labels_to_keep)), seg, 0)
    #viewer.add_labels(filtered_label_image, name='unet filtered cellpose results')

    thresholded[filtered_label_image == 0] = 0

    viewer.layers['thresholded'].data = filtered_label_image    

def on_dropdown_change(index):
   
    print(f"Selected channel: {dropdown.itemText(index)}")
    global seg_method
    layer_map = {0: 'otsu', 1: 'triangle', 2: 'isodata', 3: 'li', 4: 'mean', 5: 'minimum', 6: 'yen'}
    seg_method = layer_map[index]

def toggle_positive_nuclei_visibility():
    layer = next((layer for layer in viewer.layers if layer.name == "thresholded"), None)
    if layer:
        layer.visible = not layer.visible  

def on_file_dropdown_change(index):
    global last_path

    current_channel_name = str(dropdown_filename.itemText(index))


    layer = next((layer for layer in viewer.layers if 'channel' in layer.name), None)
    if layer:
       viewer.layers.remove(layer)

    layer = next((layer for layer in viewer.layers if layer.name == 'intensity_image'), None)
    if layer:
        viewer.layers.remove(layer)  

    layer = next((layer for layer in viewer.layers if layer.name == 'thresholded'), None)
    if layer:
        viewer.layers.remove(layer)  
    
    layer = next((layer for layer in viewer.layers if layer.name == 'intensity_image'), None)
    if layer:
        viewer.layers.remove(layer)  

    layer = next((layer for layer in viewer.layers if layer.name == 'unet_segmentation'), None)
    if layer:
        viewer.layers.remove(layer)  
    

    mask_layer = next((layer for layer in viewer.layers if layer.name == 'segmentation'), None).data

    im = imread(os.path.join(last_path, 'restitched', current_channel_name))

    thresholded = np.zeros_like(mask_layer)
    viewer.add_image(thresholded, name='thresholded', blending='additive', visible=True, colormap='red')

    stats = skimage.measure.regionprops_table(mask_layer, intensity_image=im, properties=['label', 'mean_intensity', 'area'])
    max_label = mask_layer.max()
    lookup_array = np.zeros(max_label + 1, dtype=np.float32)
    for label, mean_intensity, area in zip(stats['label'], stats['mean_intensity'], stats['area']):
            if area > size_threshold:
                lookup_array[label] = mean_intensity
    intensity_image = lookup_array[mask_layer]
    viewer.add_image(intensity_image, name='intensity_image', blending='additive', visible=False)

    viewer.add_image(im, name=current_channel_name, blending='additive', visible=True, colormap = 'green', contrast_limits=[np.amin(im), np.amax(im)])

def on_load_button_click():
    global last_path
    print("Load Button was clicked!")
    file_path = easygui.diropenbox(title="Select Processed Image Folder", default=last_path)

    if file_path is not None:
        
        last_path = file_path

        head, tail = os.path.split(file_path)

        folder_name = tail
        text_box_image_name.setText(folder_name)

        viewer.layers.clear()

        #load all files
        processed_files = os.listdir(os.path.join(file_path, 'restitched'))
        processed_files.sort()
        for file in processed_files:
            if file[-5] == str(dapi_channel):
                processed_files.remove(file)

   
 

        #delete the current unet segmentation
        dst_file =  os.path.join(file_path,  'unet_output', 'unet_seg.tif')
        if os.path.isfile(dst_file):
            os.remove(dst_file)

       # base_dir = os.path.sep.join(list(file_path.split('/')[0:-2])) 

        seg_dir = file_path + os.path.sep + 'cellpose'
        seg_files = os.listdir(seg_dir)
        seg = imread(seg_dir + os.path.sep + seg_files[0])
        viewer.add_labels(seg, name='segmentation', blending='additive', visible=False)
        dapi_im = skimage.io.imread(os.path.join(file_path, 'restitched', 'channel_' + str(dapi_channel) + '.tif'))
        viewer.add_image(dapi_im, name='DAPI', blending='additive', visible=False)




        for file in processed_files:
            dropdown_filename.addItem(file)

        #on_file_dropdown_change(0)

        #im = imread(file_path)
        
        # for name, ar in zip(files_names_list, files_list):

        #     viewer.add_image(ar, name=name, blending='additive', visible=True, colormap = 'green', contrast_limits=[np.amin(ar), np.amax(ar)])


        

   

        # seg_dir = base_dir + os.path.sep + 'cellpose'
        # 
        # seg = imread(seg_dir + os.path.sep + seg_files[0])
        # viewer.add_labels(seg, name='segmentation', blending='additive', visible=False)
        #intensity_dir = base_dir + os.path.sep + 'intensity_images'
        #intensity_image = imread(intensity_dir + os.path.sep + tail)

        # stats = skimage.measure.regionprops_table(seg, intensity_image=im, properties=['label', 'mean_intensity', 'area'])
        # max_label = seg.max()
        # lookup_array = np.zeros(max_label + 1, dtype=np.float32)
        # for label, mean_intensity, area in zip(stats['label'], stats['mean_intensity'], stats['area']):
        #         if area > size_threshold:
        #             lookup_array[label] = mean_intensity
        # intensity_image = lookup_array[seg]
        # viewer.add_image(intensity_image, name='intensity_image', blending='additive', visible=False)
        # thresholded = np.zeros_like(intensity_image)
        # viewer.add_image(thresholded, name='thresholded', blending='additive', visible=True, colormap='red')
        
        # dapi_im = skimage.io.imread(os.path.join(base_dir, 'restitched', 'channel_' + str(dapi_channel) + '.tif'))
        # viewer.add_image(dapi_im, name='DAPI', blending='additive', visible=False)

        else:
            print("No file was selected.")

        
       
    else:
        print("No file was selected.")

def calculate_threshold(intensity_im, seg_method):  
    dist = np.unique(intensity_im)
    if len(dist) % 2 != 0:
        temp = dist[:-1]
        twoD = np.array(temp).reshape(-1, 2)
    else:
        twoD = np.array(dist).reshape(-1, 2)

    
    if seg_method == 'otsu':
        thresh = skimage.filters.threshold_otsu(twoD)
    elif seg_method == 'triangle':
        thresh = skimage.filters.threshold_triangle(twoD)
    elif seg_method == 'isodata':
        thresh = skimage.filters.threshold_isodata(twoD)
    elif seg_method == 'li':
        thresh = skimage.filters.threshold_li(twoD)
    elif seg_method == 'mean':
        thresh = skimage.filters.threshold_mean(twoD)
    elif seg_method == 'minimum':
        thresh = skimage.filters.threshold_minimum(twoD)
    elif seg_method == 'yen':
        thresh = skimage.filters.threshold_yen(twoD)
    else:
        print('No method found')

    return round(thresh, 2)

def on_threshold_method_button_click():
    size_threshold = float(textbox_minsize.text()) 
    global seg_method
    scaling = float(text_box_multuplier.text())
    intensity_layer = next((layer for layer in viewer.layers if layer.name == 'intensity_image'), None).data
    mask_im = next((layer for layer in viewer.layers if layer.name == 'segmentation'), None).data

    stats = skimage.measure.regionprops_table(mask_im, intensity_image=intensity_layer, properties=['label', 'mean_intensity', 'area'])    
    filtered_stats = {key: value[stats['area'] >= size_threshold] for key, value in stats.items()}
    max_label = mask_im.max()
    lookup_array = np.zeros(max_label + 1, dtype=np.float32)
    for label, mean_intensity, area in zip(stats['label'], stats['mean_intensity'], stats['area']):
        if area > size_threshold:
            lookup_array[label] = mean_intensity
    intensity_image = lookup_array[mask_im]
    thresh = calculate_threshold(intensity_image, seg_method)
    thresh = thresh * scaling
    positive = np.zeros(intensity_image.shape, dtype=np.uint8)
    positive = np.where(intensity_image > thresh, 1, 0)
    viewer.layers['thresholded'].data = positive
    # global seg_method
    # global dist
    # size_threshold = float(textbox_minsize.text()) 
    # seg_method = dropdown.currentText()
    # multiplier = float(text_box_multuplier.text())
    # print('Segmenting with ' + seg_method + ' and multiplier ' + str(multiplier))
    
    

    # intensity_layer = next((layer for layer in viewer.layers if layer.name == 'intensity_image'), None).data
    

    # viewer.layers['thresholded'].data = np.where(intensity_layer > 0, 0, 0)
    # thresh = calculate_threshold(intensity_layer, seg_method)
    # print('using raw threshold: ' + str(thresh) + ' and multiplier: ' + str(multiplier) + ' (final = ' + str(thresh*multiplier) + ')')
    # text_box_thresh.setText(str(round(thresh*multiplier, 2)))

    # stats = skimage.measure.regionprops_table(intensity_layer, properties=['label', 'mean_intensity', 'area'])    
    # filtered_stats = {key: value[stats['area'] >= size_threshold] for key, value in stats.items()}
    #viewer.layers['thresholded'].data = np.where(intensity_layer > thresh*multiplier, 1, 0)

def on_slider_change(value):
    global global_multiplier
    real_value = value / 20.0
    text_box.setText(f"{real_value:.2f}")
    global_multiplier = real_value

def load_images(base_path, filename, file):
    
    seg = skimage.io.imread(os.path.join(base_path, 'cellpose', 'masks_' + filename + '.tif'))
    measure = skimage.io.imread(os.path.join(base_path, 'restitched', file))
    return seg, measure

def threshold_channel(seg_method, mask_im, intensity_im, scaling, size_threshold, base_path, folder, file):

    stats = skimage.measure.regionprops_table(mask_im, intensity_image=intensity_im, properties=['label', 'mean_intensity', 'area'])    
    filtered_stats = {key: value[stats['area'] >= size_threshold] for key, value in stats.items()}
    max_label = mask_im.max()
    lookup_array = np.zeros(max_label + 1, dtype=np.float32)
    for label, mean_intensity, area in zip(stats['label'], stats['mean_intensity'], stats['area']):
        if area >= size_threshold:
            lookup_array[label] = mean_intensity
    intensity_image = lookup_array[mask_im]
    skimage.io.imsave(os.path.join(base_path, folder, 'intensity_images', file), intensity_image.astype(np.uint16), check_contrast=False)
    thresh = calculate_threshold(intensity_image, seg_method)
    thresh = thresh * scaling
    positive = np.zeros(intensity_image.shape, dtype=np.uint8)
    positive = np.where(intensity_image > thresh, 1, 0)
    skimage.io.imsave(os.path.join(base_path, folder, 'positive_cells', file), positive.astype(np.uint8), check_contrast=False)

    rounded_intensity = [ '%.2f' % elem for elem in filtered_stats['mean_intensity'] ]
    
    classification = []

    for intensity in rounded_intensity:
        if float(intensity) > thresh:
            classification.append(1)
        else:
            classification.append(0)
    
    return rounded_intensity, classification, filtered_stats['label'], thresh
    
def on_apply_button_click():


    folder_path = easygui.diropenbox(title="Select Processed Image Folder")

    folders_to_process = os.listdir(folder_path)
    print('folder_path: ' + str(folder_path))
    
    
    ch1_seg_method = dropdown_ch1.currentText()
    ch2_seg_method = dropdown_ch2.currentText()
    ch3_seg_method = dropdown_ch3.currentText()
    ch1_scaling = float(textbox_scaling_ch1.text())
    ch2_scaling = float(textbox_scaling_ch2.text())
    ch3_scaling = float(textbox_scaling_ch3.text())

    size_threshold = float(textbox_minsize.text())
    
    print('ch1: ' + ch1_seg_method + ' with scaling of ' + str(ch1_scaling))
    print('ch2: ' + ch2_seg_method + ' with scaling of ' + str(ch2_scaling))
    print('ch3: ' + ch3_seg_method + ' with scaling of ' + str(ch3_scaling))
    


    
    for folder in folders_to_process:
        print('============================================')
        print('processing file: ' + str(folder))
        print('============================================')
        
        os.makedirs(os.path.join(folder_path, folder, 'positive_cells'), exist_ok=True)
        os.makedirs(os.path.join(folder_path, folder,'quantification'), exist_ok=True)
        os.makedirs(os.path.join(folder_path, folder,'organoid_masks'), exist_ok=True)

    

        df = pd.DataFrame()
        df_summary = pd.DataFrame()


        summary_labels = ['filename', 'ch2_threshold_method', 'ch3_threshold_method', 'ch4_threshold_method', 'ch2_threshold_scaling', 'ch3_threshold_scaling', 'ch4_threshold_scaling', 'total_cells']
        for i in range(2, 5):
            summary_labels.append('ch' + str(i) + '_threshold') 
            summary_labels.append('ch' + str(i) + '_positive')
            summary_labels.append('ch' + str(i) + '_unet')
            

        summary_data = [folder, ch1_seg_method, ch2_seg_method, ch3_seg_method, ch1_scaling, ch2_scaling, ch3_scaling]


        #get the mask for dapi channel
        dapi_im = skimage.io.imread(os.path.join(folder_path, folder, 'restitched', 'channel_' + str(dapi_channel) + '.tif'))
        dapi_downscaled = skimage.transform.rescale(dapi_im, 0.1, anti_aliasing=True)
        dapi_downscale_blur = skimage.filters.gaussian(dapi_downscaled, 1)
        thresh = skimage.filters.threshold_triangle(dapi_downscale_blur)
        binary = dapi_downscale_blur > thresh  
        mask_upscaled = skimage.transform.resize(binary, dapi_im.shape, anti_aliasing=False)
        skimage.io.imsave(os.path.join(folder_path, folder, 'organoid_masks', 'mask_' + folder + '.tif'), mask_upscaled.astype(np.uint8), check_contrast=False)
 

        channels_to_process = os.listdir(os.path.join(folder_path, folder, 'restitched'))
        channels_to_process.sort()

        for file in channels_to_process:
            unet_seg_bool = False
            if str(dapi_channel) not in file:
                print('processing: ' + str(file))
                
                if int(file[-5]) == 2:
                    seg_method = ch1_seg_method
                    scaling = ch1_scaling
                    if dropdown_ch1_unet.currentIndex() != 0:
                        unet_seg_bool = True
                        unet_model =  dropdown_ch1_unet.currentText()
                        unet_model_path = os.path.join('checkpoints', unet_model)
                        unet_threshold = float(text_box_unet_threshold_ch1.text())
                        print('using U-net on channel 1')
                if int(file[-5]) == 3:
                    seg_method = ch2_seg_method
                    scaling = ch2_scaling
                    # if dropdown_ch2_unet.currentIndex() != 0:
                    #     unet_seg_bool = True
                    #     unet_model =  dropdown_ch2_unet.currentText()
                    #     unet_model_path = os.path.join('checkpoints', unet_model)
                    #     unet_threshold = float(text_box_unet_threshold_ch2.text())
                    #     print('using U-net on channel 2')

                if int(file[-5]) == 4:
                    seg_method = ch3_seg_method
                    scaling = ch3_scaling
                    # if dropdown_ch3_unet.currentIndex() != 0:
                    #     unet_seg_bool = True
                    #     unet_model =  dropdown_ch3_unet.currentText()
                    #     unet_model_path = os.path.join('checkpoints', unet_model)
                    #     unet_threshold = float(text_box_unet_threshold_ch2.text())
                    #     print('using U-net on channel 3')

                seg_im, measure_im = load_images(os.path.join(folder_path, folder), folder, file)
                rounded_intensity, classification, labels, thresh = threshold_channel(seg_method, seg_im, measure_im, scaling, size_threshold, folder_path, folder, file)

                if unet_seg_bool:
                # if unet needs to be run
                    input_path = os.path.join(folder_path, folder, 'unet_temp')
                    output_path = os.path.join(folder_path, folder, 'unet_output')
                    fake_gt_path = os.path.join(folder_path, folder, 'fake_gt')
                    os.makedirs(input_path, exist_ok=True)
                    os.makedirs(output_path, exist_ok=True)
                    os.makedirs(fake_gt_path, exist_ok=True)

                    #load the initial thresholded image
                    thresholded = skimage.io.imread(os.path.join(folder_path, folder, 'positive_cells', file))
                    #rename the file to 'before unet
                    os.rename(os.path.join(folder_path, folder, 'positive_cells', file), os.path.join(folder_path, folder, 'positive_cells', file + '_before_unet.tif'))

                    yaml_file = unet_model_path[:-4] + '.yaml'
                    print('model path: ' + str(unet_model_path))
                    print('yaml path: ' + str(yaml_file))



                    skimage.io.imsave(os.path.join(folder_path, folder, 'unet_temp', 'raw_temp.tif'), measure_im, check_contrast=False)
                    yaml_file, job_name = make_config(input_path, yaml_file, unet_model_path, output_path, fake_gt_path)
                    biapy = BiaPy(yaml_file, result_dir=output_path, name=job_name, run_id=1, gpu=0)
                    biapy.run_job()

                    output_file = os.path.join(output_path, unet_model[:-4], 'results', unet_model[:-4] + '_1', 'per_image_binarized', 'raw_temp.tif')
                    dst_file =  os.path.join(output_path, 'unet_seg.tif')

                    shutil.copy(output_file, dst_file)

                    unet_seg_im = skimage.io.imread(dst_file)
                    unet_seg_im[unet_seg_im == 2] = 0
                    props = skimage.measure.regionprops_table(seg_im, intensity_image=unet_seg_im, properties=['label', 'mean_intensity'])
                    label_to_mean_intensity = {label: mean_intensity for label, mean_intensity in zip(props['label'], props['mean_intensity'])}
                    labels_to_keep = {label for label, mean_intensity in label_to_mean_intensity.items() if mean_intensity > unet_threshold}
                    filtered_label_image = np.where(np.isin(seg_im, list(labels_to_keep)), seg_im, 0)
                    thresholded[filtered_label_image == 0] = 0
                    
                    for i, label in enumerate(labels):
                        if label not in labels_to_keep:
                            classification[i] = 0

                    skimage.io.imsave(os.path.join(folder_path, folder, 'positive_cells', file), thresholded.astype(np.uint8), check_contrast=False)

                if file == 'channel_2.tif':
                    num_cells = np.amax(labels)
                    summary_data.append(num_cells)
                df['labels'] = labels
                df[file + '_intensities'] = rounded_intensity
                df[file[:-4] + '_positive'] = classification
                
                #ch2+3', 'ch3+4', 'ch2+4', 'ch2+3+4'

                
                summary_data.append(round(thresh, 2))
                summary_data.append(sum(classification))
                summary_data.append(unet_seg_bool)


    
        binary_columns = ['channel_2_positive', 'channel_3_positive', 'channel_4_positive']
        combinations = identify_positive_combinations(df, binary_columns)

        print(combinations)

        summary_labels.append('ch2+3')
        summary_labels.append('ch2+4')
        summary_labels.append('ch3+4')
        summary_labels.append('ch2+3+4')

        summary_data.append(combinations.get('channel_2+channel_3', 0))
        summary_data.append(combinations.get('channel_2+channel_4', 0))
        summary_data.append(combinations.get('channel_3+channel_4', 0))
        summary_data.append(combinations.get('channel_2+channel_3+channel_4', 0))
        df_summary['label'] = summary_labels
        df_summary['data'] = summary_data



        df.to_csv(os.path.join(folder_path, folder, 'quantification', 'quantification.csv'))
        df_summary.to_csv(os.path.join(folder_path, folder, 'quantification', 'summary.csv'))




    print('processing complete')

def identify_positive_combinations(df, binary_columns):
    """
    Identifies all combinations of positive columns in a DataFrame with binary values.
    
    Args:
        df: Pandas DataFrame containing the binary data
        binary_columns: List of column names containing binary values (0/1)
        
    Returns:
        Dictionary with combination names as keys and counts as values
    """
    # Initialize dictionary to store counts for each combination
    combination_counts = {}
    
    # Get all possible combinations (excluding empty set)
    from itertools import combinations
    all_combinations = []
    for i in range(1, len(binary_columns) + 1):
        all_combinations.extend(combinations(binary_columns, i))
    
    # Count occurrences of each combination
    for combo in all_combinations:
        combo_name = '+'.join([col.replace('_positive', '') for col in combo])
        
        # A combination is present if all columns in the combo have value 1
        combo_mask = df[list(combo)].all(axis=1)
        combo_count = combo_mask.sum()
        
        combination_counts[combo_name] = combo_count
    
    return combination_counts

def make_config(test_data_path, yaml_file, checkpoint_file, output_path, fake_gt_path):
    load_gt = False

    
    if os.path.exists( yaml_file ):
        import yaml
        with open( yaml_file, 'r') as stream:
            try:
                biapy_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        biapy_config['PATHS'] = {}
        biapy_config['PATHS']['CHECKPOINT_FILE'] = checkpoint_file
        biapy_config['MODEL'] = {}
        biapy_config['MODEL']['LOAD_CHECKPOINT'] = True

        # save file
        with open( yaml_file, 'w') as outfile:
            yaml.dump(biapy_config, outfile, default_flow_style=False)

    # Check folders before modifying the .yaml file
    if not os.path.exists(test_data_path):
        print('folder not found')
    ids = sorted(next(os.walk(test_data_path))[2])
    if len(ids) == 0:
        raise ValueError("No images found in dir {}".format(test_data_path))

    if not os.path.exists(yaml_file):
        raise ValueError("No YAML configuration file found in {}".format(yaml_file))

    if not os.path.exists(checkpoint_file):
        raise ValueError("No h5 checkpoint file found in {}".format(checkpoint_file))


    # open template configuration file
    import yaml
    with open( yaml_file, 'r') as stream:
        try:
            biapy_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    biapy_config['DATA']['TEST']['PATH'] = test_data_path
    biapy_config['DATA']['TEST']['GT_PATH'] = fake_gt_path

    biapy_config['DATA']['TEST']['LOAD_GT'] = load_gt
    biapy_config['TRAIN']['ENABLE'] = False
    biapy_config['TEST']['ENABLE'] = True
    biapy_config['MODEL']['LOAD_CHECKPOINT']= True
    biapy_config['PATHS'] = {}
    biapy_config['PATHS']['CHECKPOINT_FILE']= checkpoint_file
    biapy_config['MODEL']['N_CLASSES'] = 3

    biapy_config['TEST']['REDUCE_MEMORY'] = False

    # save file
    with open( yaml_file, 'w') as outfile:
        yaml.dump(biapy_config, outfile, default_flow_style=False)

    print( "Inference configuration finished.")

    job_name = os.path.splitext(yaml_file)[0].split('/')[-1]
    return yaml_file, job_name
 


# Create a napari viewer
viewer = napari.Viewer()

# Create a QWidget to hold buttons and sliders
widget = QWidget()
layout = QVBoxLayout()


button = QPushButton("Load an Image")
button.clicked.connect(on_load_button_click)
layout.addWidget(button)


text_box_image_name = QLineEdit()
text_box_image_name.setReadOnly(True)  
layout.addWidget(text_box_image_name)

dropdown_filename = QComboBox()
dropdown_filename.currentIndexChanged.connect(on_file_dropdown_change)
layout.addWidget(dropdown_filename)

dropdown = QComboBox()
dropdown.addItem("otsu")
dropdown.addItem("triangle")
dropdown.addItem("isodata")
dropdown.addItem("li")
dropdown.addItem("mean")
dropdown.addItem("minimum")
dropdown.addItem("yen")

dropdown.currentIndexChanged.connect(on_dropdown_change)
layout.addWidget(dropdown)



label_multiplier = QLabel("Threshold scaling factor")
layout.addWidget(label_multiplier)  # Add the label to the layout
text_box_multuplier = QLineEdit()
text_box_multuplier.setReadOnly(False)  # Make the text box read-only
text_box_multuplier.setText('1')
layout.addWidget(text_box_multuplier)


button = QPushButton("Threshold Image Using Method")
button.clicked.connect(on_threshold_method_button_click)
layout.addWidget(button)

label_thresh = QLabel("Current Threshold")
layout.addWidget(label_thresh)  # Add the label to the layout
text_box_thresh = QLineEdit()
text_box_thresh.setReadOnly(True)  # Make the text box read-only
layout.addWidget(text_box_thresh)

toggle_button = QPushButton("Toggle Positive Nuclei")
toggle_button.clicked.connect(toggle_positive_nuclei_visibility) 
layout.addWidget(toggle_button)

unet_button = QPushButton("Post-process with U-Net")
unet_button.clicked.connect(on_unet_button_click) 
layout.addWidget(unet_button)

label_unet_threshold = QLabel("Unet threshold (0-1)")
layout.addWidget(label_unet_threshold)  # Add the label to the layout
text_box_unet_threshold = QLineEdit()
text_box_unet_threshold.setReadOnly(False)  # Make the text box read-only
text_box_unet_threshold.setText('0.25')
layout.addWidget(text_box_unet_threshold)


# Set the layout on the widget and add it to the viewer
widget.setLayout(layout)
viewer.window.add_dock_widget(widget)


widget2 = QWidget()
layout2 = QVBoxLayout()



label_min_object_size = QLabel("Minimum Object Size")
layout2.addWidget(label_min_object_size) 
textbox_minsize = QLineEdit()
textbox_minsize.setReadOnly(False)  
textbox_minsize.setText('100')
layout2.addWidget(textbox_minsize)

#sep_ch1 = QLabel("--------------------------------------------------")
#layout2.addWidget(sep_ch1) 

ch1_thresh = QLabel("Channel 2 threshold method")
#label_ch3_unet = QLabel("Postprocessing with U-Net")
# layout2.addWidget(label_ch3_unet)  
# dropdown_ch3_unet = QComboBox()
# dropdown_ch3_unet.addItem("None")
# for model in available_models:
#     dropdown_ch3_unet.addItem(model)
# layout2.addWidget(dropdown_ch3_unet)

# label_unet_threshold_ch3 = QLabel("Unet threshold (0-1)")
# layout2.addWidget(label_unet_threshold_ch3)  # Add the label to the layout
# text_box_unet_threshold_ch3 = QLineEdit()
# text_box_unet_threshold_ch3.setReadOnly(False)  # Make the text box read-only
# text_box_unet_threshold_ch3.setText('0.25')
# layout2.addWidget(text_box_unet_threshold_ch3)")
layout2.addWidget(ch1_thresh) 

dropdown_ch1 = QComboBox()
dropdown_ch1.addItem("otsu")
dropdown_ch1.addItem("triangle")
dropdown_ch1.addItem("isodata")
dropdown_ch1.addItem("li")
dropdown_ch1.addItem("mean")
dropdown_ch1.addItem("minimum")
dropdown_ch1.addItem("yen")
layout2.addWidget(dropdown_ch1)

label_ch1_scaling = QLabel("Channel 2 scaling")
layout2.addWidget(label_ch1_scaling)  
textbox_scaling_ch1 = QLineEdit()
textbox_scaling_ch1.setReadOnly(False)  
textbox_scaling_ch1.setText('1')
layout2.addWidget(textbox_scaling_ch1)


label_ch1_unet = QLabel("Postprocessing with U-Net")
layout2.addWidget(label_ch1_unet)  
dropdown_ch1_unet = QComboBox()
dropdown_ch1_unet.addItem("None")
for model in available_models:
    dropdown_ch1_unet.addItem(model)
layout2.addWidget(dropdown_ch1_unet)

label_unet_threshold_ch1 = QLabel("Unet threshold (0-1)")
layout2.addWidget(label_unet_threshold_ch1)  # Add the label to the layout
text_box_unet_threshold_ch1 = QLineEdit()
text_box_unet_threshold_ch1.setReadOnly(False)  # Make the text box read-only
text_box_unet_threshold_ch1.setText('0.25')
layout2.addWidget(text_box_unet_threshold_ch1)

#sep_ch2 = QLabel("--------------------------------------------------")
#layout2.addWidget(sep_ch2) 


ch2_thresh = QLabel("Channel 3 threshold method")
layout2.addWidget(ch2_thresh) 
dropdown_ch2 = QComboBox()
dropdown_ch2.addItem("otsu")
dropdown_ch2.addItem("triangle")
dropdown_ch2.addItem("isodata")
dropdown_ch2.addItem("li")
dropdown_ch2.addItem("mean")
dropdown_ch2.addItem("minimum")
dropdown_ch2.addItem("yen")

layout2.addWidget(dropdown_ch2)


label_ch2_scaling = QLabel("Channel 3 scaling")
layout2.addWidget(label_ch2_scaling)  

textbox_scaling_ch2 = QLineEdit()
textbox_scaling_ch2.setReadOnly(False) 
textbox_scaling_ch2.setText('1')

layout2.addWidget(textbox_scaling_ch2)

# label_ch2_unet = QLabel("Postprocessing with U-Net")
# layout2.addWidget(label_ch2_unet)  
# dropdown_ch2_unet = QComboBox()
# dropdown_ch2_unet.addItem("None")
# for model in available_models:
#     dropdown_ch2_unet.addItem(model)
# layout2.addWidget(dropdown_ch2_unet)

# label_unet_threshold_ch2 = QLabel("Unet threshold (0-1)")
# layout2.addWidget(label_unet_threshold_ch2)  # Add the label to the layout
# text_box_unet_threshold_ch2 = QLineEdit()
# text_box_unet_threshold_ch2.setReadOnly(False)  # Make the text box read-only
# text_box_unet_threshold_ch2.setText('0.25')
# layout2.addWidget(text_box_unet_threshold_ch2)

#sep_ch3 = QLabel("--------------------------------------------------")
#layout2.addWidget(sep_ch3) 

ch3_thresh = QLabel("Channel 4 threshold method")
layout2.addWidget(ch3_thresh) 
dropdown_ch3 = QComboBox()
dropdown_ch3.addItem("otsu")
dropdown_ch3.addItem("triangle")
dropdown_ch3.addItem("isodata")
dropdown_ch3.addItem("li")
dropdown_ch3.addItem("mean")
dropdown_ch3.addItem("minimum")
dropdown_ch3.addItem("yen")

layout2.addWidget(dropdown_ch3)

label_ch3_scaling = QLabel("Channel 4 scaling")
layout2.addWidget(label_ch3_scaling)  

textbox_scaling_ch3 = QLineEdit()
textbox_scaling_ch3.setReadOnly(False)
textbox_scaling_ch3.setText('1')

layout2.addWidget(textbox_scaling_ch3)

# label_ch3_unet = QLabel("Postprocessing with U-Net")
# layout2.addWidget(label_ch3_unet)  
# dropdown_ch3_unet = QComboBox()
# dropdown_ch3_unet.addItem("None")
# for model in available_models:
#     dropdown_ch3_unet.addItem(model)
# layout2.addWidget(dropdown_ch3_unet)

# label_unet_threshold_ch3 = QLabel("Unet threshold (0-1)")
# layout2.addWidget(label_unet_threshold_ch3)  # Add the label to the layout
# text_box_unet_threshold_ch3 = QLineEdit()
# text_box_unet_threshold_ch3.setReadOnly(False)  # Make the text box read-only
# text_box_unet_threshold_ch3.setText('0.25')
# layout2.addWidget(text_box_unet_threshold_ch3)

sep_apply = QLabel("--------------------------------------------------")
layout2.addWidget(sep_apply) 


apply_button = QPushButton("Apply Threshold to Folder")
apply_button.clicked.connect(on_apply_button_click)



layout2.addWidget(apply_button)

widget2.setLayout(layout2)
viewer.window.add_dock_widget(widget2)



# Start the napari event loop
napari.run()

