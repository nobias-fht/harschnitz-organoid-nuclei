import napari
from qtpy.QtWidgets import QPushButton, QSlider, QVBoxLayout, QWidget, QLineEdit, QLabel, QComboBox
from qtpy.QtCore import Qt
import easygui
import numpy as np
from skimage.io import imread
import os
import pandas as pd


global global_multiplier
global ch_mean
global ch_sd
global df
global_multiplier = 0
ch_mean = 0
ch_sd = 0
df = pd.DataFrame()

def on_dropdown_change(index):
   
    print(f"Selected channel: {dropdown.itemText(index)}")
    global viewer
    global df
    global ch_mean
    global ch_sd
    layer_map = {0: 'ch_2', 1: 'ch_3', 2: 'ch_4'}
    selected_layer_name = layer_map[index]

    for layer in viewer.layers:
        layer.visible = (layer.name == selected_layer_name)
        if layer.name == selected_layer_name:
            # Clear the current selection
            viewer.layers.selection.clear()
            # Add the selected layer to the selection
            viewer.layers.selection.add(layer)
    ch_mean = df['bg_mean_' + selected_layer_name][0]
    ch_sd = df['bg_sd_' + selected_layer_name][0]

def toggle_positive_nuclei_visibility():
    layer = next((layer for layer in viewer.layers if layer.name == "positive_nuclei"), None)
    if layer:
        layer.visible = not layer.visible  

# Dummy function to be connected to the button and slider
def on_load_button_click():
    global df
    print("Load Button was clicked!")
    viewer.layers.clear()
    folder_path = easygui.diropenbox(title="Select Folder")
    if folder_path is not None:
        #get the last child in the path
        filename = os.path.basename(os.path.normpath(folder_path))
        print(filename)
        print(f"Selected folder: {folder_path}")
        ch2 = imread(folder_path + os.path.sep + 'channel_2.tif')
        ch3 = imread(folder_path + os.path.sep + 'channel_3.tif')  
        ch4 = imread(folder_path + os.path.sep + 'channel_4.tif')
        ch4_layer = viewer.add_image(ch4, name='ch_4', colormap='gray', blending='additive', visible=False, contrast_limits=(np.amin(ch4), np.amax(ch4)))
        ch3_layer = viewer.add_image(ch3, name='ch_3', colormap='gray', blending='additive', visible=False, contrast_limits=(np.amin(ch3), np.amax(ch3)))
        ch2_layer = viewer.add_image(ch2, name='ch_2', colormap='gray', blending='additive', visible=True, contrast_limits=(np.amin(ch2), np.amax(ch2)))
        
        #get everything except the last two folders from the folder_path
        normalized_path = os.path.normpath(folder_path)
        path_components = normalized_path.split(os.sep)
        path_without_last_two = os.sep.join(path_components[:-2])
        print('path without last 2: ' + str(path_without_last_two))
        df = pd.read_csv(path_without_last_two + os.path.sep + 'quantification' + os.path.sep +  filename + '_thresh.csv')

        #load the intensity images from the intensity_images folder
        ch2_intensity = imread(path_without_last_two + os.path.sep + 'intensity_images' + os.path.sep + filename + '_ch2.tif')
        ch3_intensity = imread(path_without_last_two + os.path.sep + 'intensity_images' + os.path.sep + filename + '_ch3.tif')
        ch4_intensity = imread(path_without_last_two + os.path.sep + 'intensity_images' + os.path.sep + filename + '_ch4.tif')
        viewer.add_labels(ch4_intensity, name='ch_4_intensity', blending='additive', visible=False)
        viewer.add_labels(ch3_intensity, name='ch_3_intensity', blending='additive', visible=False)
        viewer.add_labels(ch2_intensity, name='ch_2_intensity', blending='additive', visible=False)
       
        masked_data = np.zeros_like(ch2)
        temp_layer_name = "positive_nuclei"
        temp_layer = viewer.add_labels(masked_data, name=temp_layer_name, visible=True)

        viewer.layers.selection.clear()  # Clear any existing selection
        viewer.layers.selection.add(ch2_layer)  # Select ch2_layer

        on_dropdown_change(0)
    else:
        print("No folder was selected.")


def on_segment_button_click():
    global viewer 
    global global_multiplier
    global ch_mean
    global ch_sd
    # Get the currently active layer
    threshold = ch_mean + (global_multiplier * ch_sd)
    print('using mean: ' + str(ch_mean))    
    print('using sd: ' + str(ch_sd))
    print('using threshold: ' + str(threshold))
    current_layer = viewer.layers.selection.active
    if current_layer and current_layer._type_string == 'image':
        # Apply threshold
        temp_layer_name = "positive_nuclei"
        temp_layer = next((layer for layer in viewer.layers if layer.name == temp_layer_name), None)
        intensity_layer_name = current_layer.name + "_intensity"
        intensity_layer = next((layer for layer in viewer.layers if layer.name == intensity_layer_name), None)
        if intensity_layer:
            data = intensity_layer.data
            masked_data = np.where(data > threshold, 1, 0)  # Replace values below threshold with np.nan for transparency
            # Update the layer with the masked data
            if temp_layer:
            # Update the existing 'temp' layer with the new masked data
                temp_layer.data = masked_data
                temp_layer.visible = True
            else:
            # Add a new layer called 'temp' with the masked data
                temp_layer = viewer.add_labels(masked_data, name=temp_layer_name, visible=True)
        else:
            print(f"No intensity layer found for {current_layer.name}.")
    else:
        print("No active image layer selected.")

def on_slider_change(value):
    global global_multiplier
    real_value = value / 20.0
    text_box.setText(f"{real_value:.2f}")
    global_multiplier = real_value

# Create a napari viewer
viewer = napari.Viewer()

# Create a QWidget to hold buttons and sliders
widget = QWidget()
layout = QVBoxLayout()


 

# Create a button and connect it to dummy_function
button = QPushButton("Load an Image")
button.clicked.connect(on_load_button_click)
layout.addWidget(button)

dropdown = QComboBox()
dropdown.addItem("channel 2")
dropdown.addItem("channel 3")
dropdown.addItem("channel 4")
dropdown.currentIndexChanged.connect(on_dropdown_change)
layout.addWidget(dropdown)

label = QLabel("Current SD Multiplier")
layout.addWidget(label)  # Add the label to the layout

text_box = QLineEdit()
text_box.setReadOnly(True)  # Make the text box read-only
layout.addWidget(text_box)

# Create a slider and connect its value changed signal to on_slider_change function
slider = QSlider(Qt.Horizontal)
slider.setMinimum(0)
slider.setMinimum(0)  # 0 * 20 = 0
slider.setMaximum(200)  # 10 * 20 = 200
slider.setSingleStep(1)  # 0.05 * 20 = 1
slider.setValue(20)
on_slider_change(20)
slider.valueChanged.connect(on_slider_change)
layout.addWidget(slider)

# Create a button and connect it to dummy_function
button = QPushButton("Apply new threshold")
button.clicked.connect(on_segment_button_click)
layout.addWidget(button)

toggle_button = QPushButton("Toggle Positive Nuclei")
toggle_button.clicked.connect(toggle_positive_nuclei_visibility) 
layout.addWidget(toggle_button)

# Set the layout on the widget and add it to the viewer
widget.setLayout(layout)
viewer.window.add_dock_widget(widget)

# Start the napari event loop
napari.run()