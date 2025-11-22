results: folder where all results from the are saved in

all_models_sciprt: main script that is run on the HCP
- a few paths may have to be changed so it runs on your HCP
Possible run:
- python /zhome/9c/f/221532/Deep_Learning_Project/All_models_combined/all_models_script.py --category hazelnut --few-shot --n-shots 1 --save-visualizations --visualize-samples 3 --use-sam
- python /zhome/9c/f/221532/Deep_Learning_Project/All_models_combined/all_models_script.py --category hazelnut --few-shot --shots-array "5,10,15,20,25,30" --save-visualizations --visualize-samples 5
- n-62-11-12(s251763) $ python /zhome/9c/f/221532/Deep_Learning_Project/All_models_combined/all_models_script.py --category hazelnut --save-visualizations --visualize-samples 3

Remarks: 
if flag --use_sam is not enabled, than it will by default not use it
if flag --few-shot is not enabled it will do zero-shot which does not work at the moment


Analyis_and_Plots.py: contains functions to plot saved ouputs of all_models_script.py

embedding_extractor.py: calculates the similarity/anomaly scores

FOlder structure needed:

- mvtec_ad
- All_models_combined
-- results
------ posiitble_report_vis
------- all python scrips 


