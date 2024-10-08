# Introduction

Pipeline for segmenting and quantifying nuclear marker expression in organoid slices.
Written by Damian Dalle Nogare at the BioImage Analysis Infrastruture Unit of the National Facility for Data Handling and Analysis at Human Technopole, Milan. Licenced under BSD-3.

# Installing the pipeline

1. Copy the contents of the pipeline to a folder (you can pull the latest version into a git repository by using the command `git pull https://github.com/nobias-fht/harschnitz-organoid-nuclei`)
2. In the terminal, navigate to that folder
3. Create a conda environment by typing
`conda env create -f environment.yml`
4. In Fiji, go to Help → Update and then click “Manage Update Sites”
5. Add the sites: `BaSiC` and `Labkit`,  `IJBP-Plugins`, and `Local Z Projector`
6. Move the file “local_z.py” into the Fiji folder, under the `scripts/plugins` folder. If this folder does not exist, you may have to make it.

# Running the pipeline

Before running, ensure that you have the latest version of the script by running the terminal command "git pull" from the folder you have the scripts installed in.

1. Renaming the images and converting from nd2 to tif
    1. Open an anaconda terminal and navigate to where you stored the pipeline
    2. Activate the environment by typing `conda activate harschnitz_pipeline`
    3. Create a folder to store the pre-processed images by typing `mkdir folder_name` (ie `mkdir step_0_output`)
    4. run the file `step_0_rename_files.py` by typing `python3 step_0_rename_files.py`
    5. When prompted, select the location of the nd2 files, and then the output folder you made above
2. Pre-processing the data
    1. Open the script “step_1_preprocessing.ijm by dragging it into Fiji
    2. Run the script and fill in the parameters 
        1. Select Folder that the processed images from the previous step are in
        2. Choose a folder to store the output images in (important: this will be used in the following steps)
        3. For the `labkit classifier`, navigate to the ‘models’ folder and select `whole_slide_classifier.classifier`
        
3. Segmentation and Quantification
4. Confirm that the conda environment is activated. If not activate it by typing `conda activate harschnitz_pipeline`
5. Open the `config.yaml` file
    1. Update the `channel_names` entry
    2. Update the `weighting` entry (if necessary)
    3. Update the `channels` entry (if necessary)
6. Go back to the terminal
7. Make an output folder to store the results in by typing `mkdir folder_name` (ie `mkdir step_2_output)`
8. Run the script postprocessing and segmentation step by typing `python3 step_2_postprocesing_and_segmentation.py` into the terminal
9. Follow the prompts to select the output folder from step one, and the input folder you just made
10. Checking the results
    1. Check files and thresholds using  `python3 check_thresholds.py`
    2. Click the `Load an Image` button, and open the `restitched` folder. Within this folder, open a folder containing the image to check and select OK
    3. Select a channel using the dropdown, and click the `apply new threshold` button  
    4. Check the predictions of the pipeline
    5. If necessary, adjust the appropriate weight using the slider
    6. Repeat as necessary for other channels and other files (opening a new file using the `Load and Image` button when needed)
    7. Update the weights in the `config.yaml` file when finished
11. Quantify the data
    1. Run the quantification by typing `python3 step_3_quantification.py` 
    2. When prompted, select the output folder from step 2
    3. Within the output folder, there will be two new folders
        1. `report` will contain summary `csv` files that contain the results of the analysis
        2. `positive_cell_images` will contain a single binary image for each image and channel, where the positive cells are marked
