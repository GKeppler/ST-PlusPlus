# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import os
import random
from os import listdir
from os.path import isfile, join
import yaml

# %%
dataset = r"melanoma"
val_split = 1/10
splits = ["1/8", "1/4", "1/30"]
path = (r"C:\Users\Gustav\source\repos\Masterarbeit\Datasets\%s" % (dataset))
filelist = []
csv = True
csv_name = "ISIC-2017_Validation_Part3_GroundTruth.csv"

if csv:
    csv_df = pd.read_csv(os.path.join(path, csv_name), delimiter = ",", sep=",")
    for row in csv_df.iterrows():
        filelist.append("JPEGImages/%s.jpg labels/%s_segmentation.png"% (row[1]["image_id"], row[1]["image_id"]))
else:
    filelist = ["JPEGImages/%s.jpg SegmentationClass/%s"%(f[:-4],f) for f in listdir(path) if isfile(join(path, f))]

print(filelist[:2])
list_len = len(filelist) 
# %%
yaml_dict = {}
for split in splits:
    random.shuffle(filelist)
    labeled_splitpoint = int(list_len*(1-float(eval(split)))*(1-val_split))
    val_splitpoint = int(list_len*(1-val_split))
    print(f'splitpoint for {split} in dataset with list_len {list_len} are {labeled_splitpoint} and {val_splitpoint}')
    unlabeled_data = filelist[:labeled_splitpoint]
    labeled_data = filelist[labeled_splitpoint:val_splitpoint]
    val_data = filelist[val_splitpoint:]
    yaml_dict[str(split)] = dict(
        unlabeled=unlabeled_data,
        labeled=labeled_data,
        val=val_data)
with open(f'dataset\splits\{dataset}\splits.yml', 'w+') as outfile:
    yaml.dump(yaml_dict, outfile, default_flow_style=False)



# %%
text = ""
for el in val_data:
    text += el+"\n"
with open(r'dataset\splits\%s\val.txt'% dataset, 'w') as f:
    f.write(text)
text = ""
for el in unlabeled_data:
    text += el+"\n"
with open(r'dataset\splits\%s\1_8\split_0\unlabeled.txt'% dataset, 'w') as f:
    f.write(text)    
text = ""
for el in labeled_data:
    text += el+"\n"
with open(r'dataset\splits\%s\1_8\split_0\labeled.txt'% dataset, 'w') as f:
    f.write(text)    


# %%
