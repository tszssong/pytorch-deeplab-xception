import sys, os
import json
load_img_list = []
num_chair = 0
chair_anno = []

with open('./instances_train2017.json', 'r') as load_f:
    load_dict = json.load(load_f)
    for key in load_dict:
        print(key)
        if key=='images':
            load_img_list = load_dict[key]
            print('num of images:%d'%len(load_img_list))
        if key=='annotations':
            anno_list = load_dict[key]
            print('num of annotations:%d'%len(anno_list))
            for annoline in anno_list:
                for a_key in annoline:
                    if a_key == 'category_id':
                        if(annoline[a_key]==62):
                            num_chair += 1
                            chair_anno.append(annoline)
print (num_chair)
print (len(chair_anno))

chair_img = []
for annolist in chair_anno:
    id = annolist['image_id']
    for imglist in load_img_list:
        if imglist['id'] == id:
            chair_img.append(imglist)
print(len(chair_img))

chair_dict = { "info": load_dict["info"], 
               "licenses": load_dict["licenses"], 
               "images": chair_img, 
               "annotations": chair_anno, 
               "categories": load_dict["categories"] }
               #"categories": [{"supercategory":"furniture","id":62,"name":"chair"}] }
for key in chair_dict:
    print ( key, len(chair_dict[key]) )
with open("./chair_2017.json", "w") as f:
    json.dump(chair_dict, f)
print("processed ok!")
