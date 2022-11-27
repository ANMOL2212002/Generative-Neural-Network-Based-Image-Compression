from celebA import celebA
import torch
import inquirer
import datetime
import os
from analysis import *
questions = [
    inquirer.List('dataset',
                  message="Which dataset do you want to use?",
                  choices=['CIFAR', 'CelebA'],
                  ),
    # inquirer.Text('image',
    #               message="Enter the image file name",
    #               ),
]
answers = inquirer.prompt(questions)

answers['image'] = "image_1.png"
# Define a folder wrt time and date to save the images
path = "../57/" + str(datetime.datetime.now())
print(path)
answers['save_path'] = path

# Create the folder if it doesn't exist
os.makedirs(path, exist_ok=True)
# Check if image file exists
try:
    open(answers['image'])
except FileNotFoundError:
    print("File not found")
    exit()

if answers['dataset'] == 'CelebA':
    model = celebA(answers)

analysis(path+"/original/image_.png", compressed_folder=path+"/compressed/",
         output_folder=path + '/analysis/', GAN_folder=path + '/GAN/')