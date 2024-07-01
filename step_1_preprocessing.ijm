//script developed by Damian Dalle Nogare at the Human Technopole Image Analysis Facility
//released under BSD-3 License, 2024

#@ File (label="Select image folder", style="directory") imgDirectory
#@ File (label="Choose folder where to save images", style="directory") saveDirectory
#@ File (label = "LabKit classifier", style = "file") classifier


#@ Float (label="Overlap percentage", value=0.07, min=0, max=1) tileOverlap
#@ Integer (label="Camera width in px (used for computing number of tiles)", value=1600) cameraWidth
#@ Integer (label="Camera height in px (used for computing number of tiles)", value=1580) cameraHeight
/////////////////////////
// Functions
/////////////////////////
function stripWhiteSpaces(string) {
	newString = replace(string, " ", "");
	return newString;
}



fileExtension = ".tif";

//setBatchMode(true);
savePath_localz = saveDirectory+File.separator+"local_z";
if(!File.exists(savePath_localz)) {File.makeDirectory(savePath_localz);};
savePath_sum = saveDirectory+File.separator+"sum_slices";
if(!File.exists(savePath_sum)) {File.makeDirectory(savePath_sum);};


file_list = getFileList(imgDirectory);
number_of_files = file_list.length;

for (i=0; i< number_of_files; i++) {

	if (!File.exists(saveDirectory + File.separator + "sum_slices" + File.separator + file_list[i])) {
	
		//LOCAL Z PROJECTION
		print("running image " + i+1 + " of " + number_of_files + ": " + file_list[i]);
	 	open(imgDirectory + File.separator + file_list[i]);
	 	
	 	imageName = getTitle();
	    imageTitle = substring(imageName, 0, indexOf(imageName, fileExtension));
	    imageTitle = stripWhiteSpaces(imageTitle);
	 	savePath = saveDirectory+File.separator+imageTitle;
	    if(!File.exists(savePath)) {File.makeDirectory(savePath);};
	 	
	 	getRawStatistics(count, mean, min, max, std);
	 	if (min == 0) {
	 		run("Replace/Remove Label(s)", "label(s)=0 final=100");
	 	}
	    imageName = getTitle();
	    imageTitle = substring(imageName, 0, indexOf(imageName, fileExtension));
	    imageTitle = stripWhiteSpaces(imageTitle);
	    run("local z");
	    saveAs("Tiff", savePath_localz + File.separator  + file_list[i]);
		run("Duplicate...", "use");
	    //selectWindow("1");
	    run("Scale...", "x=.25 y=.25 width=1121 height=1107 interpolation=Bilinear average create title=down");
	    run("Segment Image With Labkit", "input=down segmenter_file=" + classifier + " use_gpu=true");
	    run("Scale...", "x=4 y=4 width=4484 height=4428 interpolation=None create");
	    save(savePath + File.separator + "slice_mask.tif");
	    selectWindow(imageName);
	    run("Z Project...", "projection=[Sum Slices]");
	    run("16-bit");
	    run("Subtract Background...", "rolling=100 stack");
	    save(savePath_sum + File.separator + file_list[i]);
	    
	    close("*");
 	} else {
		print("File already exists, skipping");
	}
}
print("Finished pre-processing, starting flatfield estimation");

imgDirectory = savePath_localz;

//print(imgDirectory);

//if(lengthOf(imgDirectory) == 0 || lengthOf(saveDirectory) == 0 || lengthOf(fileExtension) < 3) {exit("Missing arguments");};

// if(numberXTiles <= 1 && numberYTiles <= 1) {exit("Either X or Y need more tiles, both cannot be 1 or less");};

if(cameraWidth < 512 || cameraHeight < 512) {exit("Camera Width or Height set too low");};

fileList = getFileList(imgDirectory);
if(fileList.length == 0){exit("Image Folder appears to be empty");};

setBatchMode(true);
call("java.lang.System.gc");
	
for(fileIdx = 0; fileIdx < fileList.length; fileIdx++){
	filename_only = substring(fileList[fileIdx], 0, lastIndexOf(fileList[fileIdx] , '.'));
	print(saveDirectory + File.separator + filename_only + File.separator + filename_only + ".tif_ch4.tif");
	print(File.exists(saveDirectory + File.separator + filename_only + File.separator + filename_only + ".tif_ch4.tif"));
	if (!File.exists(saveDirectory + File.separator + filename_only + File.separator + filename_only + ".tif_ch4.tif")) {
	
		for (fileset = 0; fileset < 2; fileset++) {  //looop through both the localz and the sum projections
		    showProgress(fileIdx, fileList.length);
		    if(endsWith(fileList[fileIdx], fileExtension)) {
		        //open file with Bio Formats
		        if (fileset == 0) {
		        	print("running " + fileList[fileIdx] + " local_z");
			        run("Bio-Formats Windowless Importer", "open=["+imgDirectory+File.separator+fileList[fileIdx]+"]");
					ymlsavePath = saveDirectory+File.separator+filename_only;
			        if File.exists(ymlsavePath+File.separator+"data.yml") {
			        	File.delete(ymlsavePath+File.separator+"data.yml");
			        }

			        file = File.open(ymlsavePath+File.separator+"data.yml");
		        } else {
		       		print("running " + fileList[fileIdx] + " max projections");
		        	run("Bio-Formats Windowless Importer", "open=["+savePath_sum+File.separator+fileList[fileIdx]+"]");
		        }
		        imageName = getTitle();
		        imageTitle = substring(imageName, 0, indexOf(imageName, fileExtension));
		        imageTitle = stripWhiteSpaces(imageTitle);
		        Stack.getDimensions(width, height, channels, slices, frames);
		          
		        
		        //create local dir to save files for final stitching
		        savePath = saveDirectory+File.separator+imageTitle;
		        if(!File.exists(savePath)) {File.makeDirectory(savePath);};
		        
		        //getting number of tiles and overlap
		        xTilesNumber = Math.ceil(width / cameraWidth);
		        yTilesNumber = Math.ceil(height / cameraHeight);
		        tileSizeX = parseInt(width / xTilesNumber);
		        tileSizeY = parseInt(height / yTilesNumber);
		        realCameraWidth = tileSizeX * (1 + tileOverlap);
		        realCameraHeight = tileSizeY * (1 + tileOverlap);
		        
		        
				if (fileset == 0) {
					print(file, "XTiles: "+ xTilesNumber +"\n");
					print(file, "YTiles: "+ yTilesNumber +"\n");
					print(file, "XOverlap:  "+tileSizeX * (1 + tileOverlap) - tileSizeX);
					print(file, "YOverlap:  "+tileSizeY * (1 + tileOverlap) - tileSizeY);
				}
		         // LOOP THROUGH ALL CHANNELS HERE
		         //duplicate a channel
		         //use that imgae
		         
		         
		         if (fileset == 0) {
		         	start_channel = 1;
		         	end_channel = 1;
		         } else {
		         	start_channel = 2;
		         	end_channel = 4;
		         }
		         
		         
		         for (i = start_channel; i<=end_channel; i++) {
		         	
					selectWindow(imageName);
					close("\\Others");
					run("Duplicate...", "duplicate channels=" + i);
		         	chanName = getTitle();
		         
		        
			        
			        //Flat and Dark field estimation for each Z starts here
			        for(z = 1; z <= slices; z++){
			            selectWindow(chanName);
			            Stack.setSlice(z);
			            tile_pos = 0;
			            for(y = 0; y < yTilesNumber; y++){
			                for(x = 0; x < xTilesNumber; x++){
			                    selectWindow(chanName);
			                    x_coord = (x * tileSizeX) - (tileSizeX * (tileOverlap / 2));
			                    y_coord = (y * tileSizeY) - (tileSizeY * (tileOverlap / 2));
			                    if(x_coord < 0) {x_coord = 0;};
			                    if(y_coord < 0) {y_coord = 0;};
			                    if(x_coord + realCameraWidth > width){ x_coord = width - realCameraWidth; };
			                    if(y_coord + realCameraHeight > height) { y_coord = height - realCameraHeight; };
			                    run("Specify...", "width="+realCameraWidth+" height="+realCameraHeight+" x="+x_coord+" y="+y_coord);
			                    
			                     if (fileset == 0) {
			                    	
			                    	print(file, "tile_" + tile_pos + "_xpos: " + x_coord);
			                    	print(file, "tile_" + tile_pos + "_ypos: " + y_coord);
			                    	tile_pos = tile_pos + 1;
			                    }
			                    	
			                    
			                    run("Duplicate...", "title=["+imageTitle+"_z"+z+"_y"+y+"_x"+x+"]");
			                };
			            };
			            toCorrect = "temp_stack";
			            run("Images to Stack", "name=["+toCorrect+"] title=_x");
			
			            estimationStack = toCorrect+"_noempty_slices";
			            run("Duplicate...", "title=["+estimationStack+"] duplicate");
			
			            selectWindow(estimationStack);
			            Stack.getDimensions(tempWidth, tempHeight, tempChannels, tempSlices, tempFrames);
			            //for(z2 = tempSlices; z2 >= 1; z2--){
			            //    Stack.setSlice(z2);
			            //    List.setMeasurements;
			            //    minIntensity = List.getValue("Min");
			                //if(minIntensity == 0) {run("Delete Slice");};
			            //};
			
			            // this estimates Flat and Dark fields, removes frames with empty corners or 0 (zero) pixels
			            selectWindow(estimationStack);
			            run("BaSiC ", "processing_stack="+estimationStack+" flat-field=None dark-field=None shading_estimation=[Estimate shading profiles] shading_model=[Estimate both flat-field and dark-field] setting_regularisationparametes=Automatic temporal_drift=Ignore correction_options=[Compute shading only] lambda_flat=0.50 lambda_dark=0.50");
			
			            // this fixes shading of all frames
			            run("BaSiC ", "processing_stack="+toCorrect+" flat-field=Flat-field:"+estimationStack+" dark-field=Dark-field:"+estimationStack+" shading_estimation=[Skip estimation and use predefined shading profiles] shading_model=[Estimate both flat-field and dark-field] setting_regularisationparametes=Automatic temporal_drift=Ignore correction_options=[Compute shading and correct images] lambda_flat=0.50 lambda_dark=0.50");
			
			            //saves flat and dark fields images
			            shadingSavePath = savePath+File.separator+"flatfield_images";
			            if(!File.exists(shadingSavePath)){File.makeDirectory(shadingSavePath);};
			
			            selectWindow("Flat-field:"+estimationStack);
			            saveAs("Tiff", shadingSavePath+File.separator+"Flat-field:"+estimationStack+"_z"+z+".tif");
			            close("Flat-field:"+estimationStack+"_z"+z+".tif");
			
			            selectWindow("Dark-field:"+estimationStack);
			            saveAs("Tiff", shadingSavePath+File.separator+"Dark-field:"+estimationStack+"_z"+z+".tif");
			            close("Dark-field:"+estimationStack+"_z"+z+".tif");
			
			            close("Flat-field:"+toCorrect);
			            close("Dark-field:"+toCorrect);
			            close(estimationStack);
			            close(toCorrect);
			            
			            selectImage("Corrected:temp_stack");
			            rename("Slice_"+z);
			        };
			        
					run("16-bit");
					
					//somehow this is saturating now on the sum projected images
					
			        saveAs("Tiff", savePath + File.separator + imageName + "_ch" + i);
	
	
					
		
		         }
		    
				File.close(file);
		
		
		        run("Close All");
		        call("java.lang.System.gc");
		    };
		};
	}
	else {
		print("corercted file Exists, skipping");
	}
};


setBatchMode("exit and display");
//beep();
print("Macro finished sucessfuly");


    
