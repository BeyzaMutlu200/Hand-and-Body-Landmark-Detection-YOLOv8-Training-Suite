import os 
import random
import itertools
import shutil

# Define the variables
OUTPUT_FOLDER = r'dataset\SplitData'
INPUT_FOLDER = r'dataset\all'

LIST = ['train','val','test']
SUB_LIST = ['images','labels']
SPLIT_DATA = {"train":0.7,"val":0.2,"test":0.1}
CLASSES = ["RightHandUp","Piece","Three"]

# Create yaml function 
# Don't try to move the spaces
def create_yaml(classes,folder_path):
   
 dataYaml= f'path: ../Data\n\
 train: ../train/images\n\
 val: ../val/images\n\
 test: ../test/images\n\
 \n\
 nc:{len(classes)}\n\
 names:{classes}'

 f=open(f"{folder_path}/data.yaml",'a')
 f.write(dataYaml)
 f.close()
 print('data.yaml file is created')

# Create func to control folder existence
def control_folder(folder_path):
    # Control the OUTPUT_FOLDER
    exists = os.path.exists(folder_path)
    if exists is False:
        os.mkdir(folder_path)
        print("The folder path is created")

# Create subfolders for splitting 
def create_subfolders(folder_path,folder_label,sub_folder_labels):
    #folder_parth: the main path that exists
    #folder_label: OUTPUT/folder_label{train}
    #sub_folder_labels: OUTPUT/folder_label/sub_folder_label{images}-
    for l in folder_label :
      path = os.path.join(folder_path,l)
      for items in sub_folder_labels:
        sub_path = os.path.join(path,items)
        try:
         os.makedirs(sub_path)
        except:
         print("this file already exists")

# Copy the text and image files into subfolders (.txt, .jpg) 
def copy_txt_jpg_files(o_put,copy_folder,output_folder,folder_label):
    for i, subset in enumerate(o_put):
        for item in subset:
            # Copy the files
            try:
                shutil.copy(os.path.join(copy_folder, f"{item}.txt"), os.path.join(output_folder, folder_label[i], 'labels', f"{item}.txt"))
                shutil.copy(os.path.join(copy_folder, f"{item}.jpg"), os.path.join(output_folder, folder_label[i], 'images', f"{item}.jpg"))
            except FileNotFoundError as e:
                print(f"Error copying file: {e}")

def remove_ext(folder_path):
    print(f"Processing folder: {folder_path}")
    
    n_txt_list = [f[:-4] for f in os.listdir(folder_path) if f.endswith('.txt')]
    n_jpg_list = [f[:-4] for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    print(f"Number of .txt files: {len(n_txt_list)}")
    print(f"Number of .jpg files: {len(n_jpg_list)}")
    print(f"txt list: {n_txt_list}")
    print(f"jpg list: {n_jpg_list}")
    
    if n_txt_list == n_jpg_list:
        print("Lists are equal")
        unique_name_list = n_jpg_list
    else:
        print("Lists are not equal")
        # Handle the case when lists are not equal
        unique_name_list = list(set(n_txt_list + n_jpg_list))
    
    return unique_name_list
def splitting_data(unique_name_list,folder_path):
    all_list = os.listdir(folder_path)   # Define the value to split data 
    TEST_SPLIT = int(len(unique_name_list)*SPLIT_DATA['test'])
    TRAIN_SPLIT = int(len(unique_name_list)*SPLIT_DATA['train'])
    VAL_SPLIT = (len(unique_name_list)*SPLIT_DATA['val'])
    TOTAL_SPLIT = TEST_SPLIT+TRAIN_SPLIT+VAL_SPLIT

    add_missing_data=0
    if len(all_list) != TOTAL_SPLIT:
        add_missing_data = len(all_list) - TOTAL_SPLIT
        TEST_SPLIT += add_missing_data
        print("missing data added")
    # Split data 
    LENGTH_TO_SPLIT = [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT] # the usage output[0]=TRAIN , output[1]=VALIDATION , output[2]=TEST 
    random.shuffle(unique_name_list)
    input = iter(unique_name_list)
    output = [list(itertools.islice(input,int(data))) for data in LENGTH_TO_SPLIT] 
    return output

# Control the folder existence
control_folder(OUTPUT_FOLDER)

#remove extensions 
unique_name_list = remove_ext(INPUT_FOLDER)

#split the data
output = splitting_data(unique_name_list,INPUT_FOLDER)

# Create folder for splitting data 
create_subfolders(OUTPUT_FOLDER,LIST,SUB_LIST)

# Copy the text and image files into subfolders 
copy_txt_jpg_files(output,INPUT_FOLDER,OUTPUT_FOLDER,LIST)

# Create yaml file
create_yaml(CLASSES,OUTPUT_FOLDER)





