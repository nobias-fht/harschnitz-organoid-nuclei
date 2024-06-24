import os
import numpy as np
import shutil
import nd2
import tifffile
import easygui


path = easygui.diropenbox('Select folder with nd2 images')
output_folder = easygui.diropenbox('Select folder to save preprocessed images')
files = os.listdir(path)
files.sort()

for i in range(0, len(files), 2):
    filename_0 = files[i].split(' ')[0]
    filename_1 = files[i+1].split(' ')[0]
    print('loading files ' + str(files[i]) + ' and ' + str(files[i+1]) + ' iteration ' + str(i))
    im_1 = nd2.imread(path + os.path.sep +  files[i])
    im_2 = nd2.imread(path + os.path.sep +  files[i+1])
    
    if len(im_1.shape) == 3:
        assert len(im_2.shape) == 4
        assert im_1.shape[0] == im_2.shape[0]
        assert im_1.shape[1] == im_2.shape[2]
        assert im_1.shape[2] == im_2.shape[3]
        newim = np.stack((im_1, im_2[:,0,:,:], im_2[:,1,:,:], im_2[:,2,:,:]), axis=1)   
        print('saving file ' + str(filename_0 + '.tif'))     
        tifffile.imwrite(output_folder + os.path.sep + filename_0 + '.tif', newim, imagej=True)
      
                                
    if len(im_1.shape) == 4:
        assert len(im_2.shape) == 3
        assert im_2.shape[0] == im_1.shape[0]
        assert im_2.shape[1] == im_1.shape[2]
        assert im_2.shape[2] == im_1.shape[3]
        newim = np.stack((im_2, im_1[:,0,:,:], im_1[:,1,:,:], im_1[:,2,:,:]), axis=1)        
        print('saving file ' + str(filename_0 + '.tif'))     
        tifffile.imwrite(output_folder + os.path.sep + filename_0 + '.tif', newim, imagej=True)
                                    


