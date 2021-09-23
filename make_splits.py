# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import os
import random
from os import listdir
from os.path import isfile, join


# %%
dataset = r"pascal"
path = (r"C:\Users\Gustav\Documents\GitHub\Masterarbeit\Datasets\%s\SegmentationClass" % (dataset))
print(path)
filelist = []
csv = False
csv_name = "ISIC-2017_Validation_Data_metadata.csv"

if csv:
    csv_df = pd.read_csv(os.path.join(path, csv_name), delimiter = ",", sep=",")
    for row in csv_df.iterrows():
        filelist.append("JPEGImages/%s.jpg labels/%s+_segmentation.png"% (row[1]["image_id"], row[1]["image_id"]))
else:
    filelist = ["JPEGImages/%s.jpg SegmentationClass/%s"%(f[:-4],f) for f in listdir(path) if isfile(join(path, f))]

print(filelist[:2])
random.shuffle(filelist)
len = len(filelist) 
split = 1.0/8.0


unlabeled_data = filelist[:int(len*(1-split))]
labeled_data = filelist[int(len*(1-split)):-100]
val_data = filelist[-100:]


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
