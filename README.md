# Introduction

Pipeline for segmenting and quantifying nuclear marker expression in organoid slices.
Written by Damian Dalle Nogare at the BioImage Analysis Infrastruture Unit of the National Facility for Data Handling and Analysis at Human Technopole, Milan. Licenced under BSD-3.

Updated 12-5-2025

# Installing the pipeline

1. Copy the contents of the pipeline to a folder (you can pull the latest version into a git repository by using the command `git pull https://github.com/nobias-fht/harschnitz-organoid-nuclei`)
2. In the terminal, navigate to that folder
3. Create a conda environment by typing
`conda env create -f environment.yml`
4. In Fiji, go to Help → Update and then click “Manage Update Sites”
5. Add the sites: `BaSiC` and `Labkit`,  `IJBP-Plugins`, and `Local Z Projector`
6. Move the file “local_z.py” into the Fiji folder, under the `scripts/plugins` folder. If this folder does not exist, you may have to make it.
7. Deactivate the current environment by typing `conda deactivate`
8. Install BiaPy into a second environment by following the instructions [here](https://biapy.readthedocs.io/en/latest/get_started/installation.html), and the modifications below. Make sure you are using the `command line`, `Conda + pip`, and `GPU Support` tabs on that page for the correct installation instructions, and then make the alterations at the bottom of this page.


# Running the processing pipeline

Before running, ensure that you have the latest version of the script by running the terminal command "git pull" from the folder you have the scripts installed in.

1. Pre-processing the data
    1. Open the script “step_1_preprocessing.ijm by dragging it into Fiji
    2. Run the script and fill in the parameters 
        1. Select Folder that the processed images from the previous step are in
        2. Choose a folder to store the output images in (important: this will be used in the following steps)
        3. For the `labkit classifier`, navigate to the ‘models’ folder and select `whole_slide_classifier.classifier`
        
2. Segmentation and Quantification
3. Confirm that the conda environment is activated. If not activate it by typing `conda activate harschnitz_pipeline`
4. If necessary, open the `config.yaml` file and update the following entries:
    1. `channels` The channels to quantify (ie `[2,3,4]` would quantify channels 2, 3 and 4)
    2. `dapi_channel` The position of the DAPI channel in the channel stack
    3. `channel_names` The names of the channels (in the same order as the `channels` values)

5. Go back to the terminal
6. Make an output folder to store the results in by typing `mkdir folder_name` (ie `mkdir step_2_output)` (you can also make this folder using the file manager GUI in your OS)
7. Run the script postprocessing and segmentation step by typing `python3 step_2_postprocesing_and_segmentation.py` into the terminal
8. Follow the prompts to select the output folder from step one, and the input folder you just made

At the end of this step, you should have segmented images and organoid masks ready for quantification

# Checking and Quantifying the Results

1. deactivate the current environment, and activate the second (biapy) environment you made above by typing `conda activate environment_name`
2. Start the GUI by typing `python step_3_quantify_cells.py`. In the GUI there are two widgets, the top, which controls exploratory segmentation of single images, and the lower, which controls global segmentation of ALL images in a folder.
3. The first step is to decide on an appropriate threshold method to determine which cells are positive and which are negative using the exploratory widget.
    1. Open an organoid image by pressing the `Load an Image` button at the top and selecting an image output folder from the file picker. This will load all channels, and you should see the filename appear in the section below the Load button. You can select different channels using the dropdown below the filename.
4. Try different segmentation approaches to determine which one works best on a given channel.
    1. The `otsu`, `triangle`, and `mean` approaches tend to work well. 
    2. If you would like to further adjust the threshold, you can adjust the `Threshold Scaling Factor` value. This will multiply the threshold by this amount (ie a scaling factor of 2 would multiple the threshold by 2, resulting in a more stringent segmentation)
    3. Classify the current channel by pressing the `Threshold Image Using Method Button`. In the resulting screen, the red cells are counted as potiive. You can toggle these on and of by pressing the `Toggle Positive Nuclei` button. It may also be helpful to make the dapi channel visible in the layers tab. 
    4. Once you are happy with a segmentation approach and scaling factor, input those values into the second widget in the appropriate places 
    5. Repeat for all channels.

5. (Optional) U-net post processing
    1. This pipeline implements a U-net for postprocessing in the case that too many spurious detections are made due to high background. To use this on a given channel, click the `Post Process with Unet` button. A new classification image will appear that has been post processed with the U-net to attempt to remove non-nuclear detections.
    2. This can be turned on in the global settings per-channel by changing the `Postprocessing with Unet` dropdown from `None` to `Unet.pth`.

6. Quantify all of the folders 
    1. Once you are happy with all the settings, and they have been inputted into the global widget click the `Apply Threshold to Folder` button at the bottom of the global widget.
    2. Select the output folder from step 2 above, which contains all of the output folders for each image, and click OK.
    3. The pipeline will quantify all of the images  in this folder according to the settings you have chosen. 
    4. Within the output folder for each image, there will be a new folder called `quantification` containing two files.
        1. `quantification.csv` will contain the raw quantification for every cell in the image for each channel, as well as a binary value (0 or 1) determining whether or not that cell was positive (above the selected threshold) or not.
        2. `summary.csv` contains summary information about the image, such as total cells, how many cells were positive for each channel, what the thresholds used were and other information.


Update to BiaPy installation instructions

conda create -n BiaPy_env python=3.10
conda activate BiaPy_env
conda install -c conda-forge napari pyqt
pip install biapy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm pytorch-msssim torchmetrics[image]==1.4.*
pip install easygui
pip install opencv-python-headless


