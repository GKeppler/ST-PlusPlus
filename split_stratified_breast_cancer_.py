import random
from os import listdir
from os.path import isfile, join
import yaml
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# set basic params and load file list
cross_val_splits = 5
num_shuffels = 5
splits = ["1", "1/4", "1/8", "1/30"]
# /lsdf/kit/iai/projects/iai-aida/Daten_Keppler/ISIC_Demo_2017")
images_folder = "images"
labels_folder = "labels"
training_filelist = []
val_filelist = []
test_filelist = []

# pnuemothorax dataset
dataset = r"breastCancer"
base_path = r"/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/BreastCancer"
training_filelist = [
    "train/images/%s train/labels/%s_mask.png" % (f, f[:-4])
    for f in listdir(join(base_path, "train", images_folder))
    if isfile(join(base_path, "train", images_folder, f))
]
# sanity check if file in image folder are same as in
differences = set(
    [
        "train/images/%s.png train/labels/%s_mask.png" % (f[:-9], f[:-9])
        for f in listdir(join(base_path, "train", labels_folder))
        if isfile(join(base_path, "train", labels_folder, f))
    ]
).symmetric_difference(set(training_filelist))
if len(differences) != 0:
    raise Exception(
        f"files in folders '{images_folder}' and '{labels_folder}' do not match because of: {differences}"
    )

test_filelist = [
    "test/images/%s test/labels/%s_mask.png" % (f, f[:-4])
    for f in listdir(join(base_path, "test", images_folder))
    if isfile(join(base_path, "test", images_folder, f))
]

list_len = len(training_filelist)
print(training_filelist[:2])

# shuffle labeled/unlabled
for shuffle in range(num_shuffels):
    yaml_dict = {}
    for split in splits:
        random.shuffle(training_filelist)
        # calc splitpoint
        labeled_splitpoint = int(list_len * float(eval(split)))
        print(
            f"splitpoint for {split} in dataset with list_len {list_len} are {labeled_splitpoint}"
        )
        unlabeled = training_filelist[labeled_splitpoint:]
        labeled = training_filelist[:labeled_splitpoint]
        skf = StratifiedKFold(n_splits=cross_val_splits)
        y = [(0 if name[0] == "n" else 1 if name[0] == "m" else 2) for name in labeled]
        count = 0
        for train_index, val_index in skf.split(labeled, y):
            unlabeled_copy = unlabeled.copy()  # or elese it cant be reused
            train = [labeled[i] for i in train_index]
            val = [labeled[i] for i in val_index]
            yaml_dict["val_split_" + str(count)] = dict(
                unlabeled=unlabeled_copy, labeled=train, val=val
            )
            count += 1

        # save to yaml
        # e.g 1/4 -> 1_4 for folder name
        zw = list(split)
        if len(zw) > 1:
            zw[1] = "_"
        split = "".join(zw)

        yaml_path = rf"dataset/splits/{dataset}/{split}/split_{shuffle}"
        Path(yaml_path).mkdir(parents=True, exist_ok=True)
        with open(yaml_path + "/split.yaml", "w+") as outfile:
            yaml.dump(yaml_dict, outfile, default_flow_style=False)
# test yaml file
yaml_dict = {}
yaml_path = rf"dataset/splits/{dataset}/"
Path(yaml_path).mkdir(parents=True, exist_ok=True)

with open(yaml_path + "/test.yaml", "w+") as outfile:
    yaml.dump(test_filelist, outfile, default_flow_style=False)
