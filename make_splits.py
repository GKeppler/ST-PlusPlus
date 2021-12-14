# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import os
import random
from os import listdir
from os.path import isfile, join
import yaml
from pathlib import Path

# %%
dataset = r"melanoma"
val_split = 1/10
splits = ["1/8", "1/4", "1/30"]
path = (r"/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/ISIC_Demo_2017")
training_filelist = []
val_filelist = []
csv = True
if csv:
    #Training
    csv_df = pd.read_csv(os.path.join(path, "ISIC-2017_Training_Data_metadata.csv"), delimiter = ",", sep=",")
    for row in csv_df.iterrows():
        training_filelist.append("images/%s.jpg labels/%s_segmentation.png"% (row[1]["image_id"], row[1]["image_id"]))
    #Validation
    csv_df = pd.read_csv(os.path.join(path, "ISIC-2017_Validation_Data_metadata.csv"), delimiter = ",", sep=",")
    for row in csv_df.iterrows():
        val_filelist.append("images/%s.jpg labels/%s_segmentation.png"% (row[1]["image_id"], row[1]["image_id"]))
else:
    training_filelist = ["images/%s.jpg SegmentationClass/%s"%(f[:-4],f) for f in listdir(path) if isfile(join(path, f))]

list_len = len(training_filelist)
print(training_filelist[:2],list_len)

# %%
#process val
random.shuffle(training_filelist)
if csv:
     val_data = val_filelist
     training_filelist = training_filelist
else:
    val_splitpoint = int(list_len*val_split)
    val_data = filelist[:val_splitpoint]
    #remove val_files from list
    training_filelist = filelist[val_splitpoint:]
# %%
yaml_dict = {}
for split in splits:
    random.shuffle(training_filelist)
    labeled_splitpoint = int(list_len*float(eval(split)))
    print(f'splitpoint for {split} in dataset with list_len {list_len} are {labeled_splitpoint}')
    unlabeled_data = training_filelist[:labeled_splitpoint]
    labeled_data = training_filelist[labeled_splitpoint:]
    yaml_dict = dict(
        unlabeled=unlabeled_data,
        labeled=labeled_data,
        val=val_data)
    #save to yaml
    ## e.g 1/4 -> 1_4 for folder name
    zw = list(split)
    zw[1]="_"
    split = "".join(zw)
    yaml_path = fr"dataset/splits/{dataset}/{split}/split_0"
    Path(yaml_path).mkdir(parents=True, exist_ok=True)
    with open(yaml_path+'/split.yaml', 'w+') as outfile:
        yaml.dump(yaml_dict, outfile, default_flow_style=False)



# %%
# text = ""
# for el in val_data:
#     text += el+"\n"
# with open(r'dataset\splits\%s\val.txt'% dataset, 'w') as f:
#     f.write(text)
# text = ""
# for el in unlabeled_data:
#     text += el+"\n"
# with open(r'dataset\splits\%s\1_8\split_0\unlabeled.txt'% dataset, 'w') as f:
#     f.write(text)    
# text = ""
# for el in labeled_data:
#     text += el+"\n"
# with open(r'dataset\splits\%s\1_8\split_0\labeled.txt'% dataset, 'w') as f:
#     f.write(text)    
