## Autonomous Detection of Plant Disease Symptoms Directly from Aerial Imagery
Harvey Wu, Tyr Wiesner-Hanks, Ethan L. Stewart, Chad DeChant, Nicholas Kaczmar, Michael A. Gore, Rebecca J. Nelson, and Hod Lipson

Summary: using deep learning, we detect a corn disease, Northern Leaf Blight (NLB), through aerial images of corn fields captured by small UAVs.

Code organization:       
`/boom_transfer.py` trains a CNN on images stored in the directory `/new_data` (you need to create it).   
`/test.py` tests a trained CNN stored in `/models` on a test set of data in `/new_data`.   
`/boom_heatmaps.py` generates heatmaps using a trained CNN stored in `/models`.     
`scripts/crop_lesions.py` samples subimages containing lesions from a set of labeled images containing lesions.      
`scripts/crop_nonlesions.py` samples subimages without lesions from a set of labeled images without lesions.      
`scripts/yesno.py` splits the dataset specified in [1] into images with lesions and images without.       
`scripts/overlay.py` creates a composite image with heatmap overlaid on top of the original.      
`scripts/drawlines.py` draws the annotations onto the original images (semimajor axis of lesion).     
`scripts/*.pkl` contain image names; they correspond to the train/val/test split used in our experiments.        

Contact wu.harvey (at) columbia.edu with any questions.
