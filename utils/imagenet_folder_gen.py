# ImageNet 1k
import glob
import os
import shutil
import xml.dom.minidom

from tqdm import tqdm

prefix = "/data/qinyu/ImageNet_1k_2012/val"
xml_files = glob.glob("/data/qinyu/ImageNet_1k_2012/ann/val/*.xml")
xml_files = sorted(xml_files)
print(len(xml_files))

i = 0
label_set = set()
for file in tqdm(xml_files):
    dom = xml.dom.minidom.parse(file)
    root = dom.documentElement
    file_name = root.getElementsByTagName('filename')
    file_name = file_name[0].firstChild.data
    file_name = f"{file_name}.JPEG"

    label = root.getElementsByTagName('name')[0].firstChild.data
    label_set.add(label)

    folder = os.path.join(prefix, label)
    if not os.path.exists(folder):
        os.makedirs(folder)
    shutil.move(os.path.join(prefix, file_name), os.path.join(folder, file_name))
