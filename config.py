import os

# initials
# CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#            'train', 'tvmonitor']




# default
# CLASSES = ['truck',                # 0
#            'bicycle',               # 1
#            'traffic light',              # 2
#            'void',              # 3
#            'void',              # 4
#            'bus',                   # 5
#            'car',                   # 6
#            'void',                  
#            'void', 
#            'void',                  
#            'void', 
#            'void', 
#            'void',
#            'motorbike',             # 13
#            'person',                # 14
#            'void',                  # 15
#            'void', 
#            'void',
#            'train',                 # 18
#            'void']

CLASSES = ['truck',                # 0
           'car',               # 1
           'pdestrian',              # 2
           'traffic light',              # 3
           'void',              # 4
           'bus',                   # 5
           'car',                   # 6
           'void',                  
           'void', 
           'void',                  
           'void', 
           'void', 
           'void',
           'motorbike',             # 13
           'person',                # 14
           'void',                  # 15
           'void', 
           'void',
           'train',                 # 18
           'void']


BATCH_SIZE = 20
CELL_SIZE = 7
BOXES_PER_CELL = 2
IMAGE_SIZE = 448

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 1.0
COORD_SCALE = 5.0

LEARNING_RATE = 0.001
ALPHA = 0.1
DISP_CONSOLE = True

MAX_ITER = 120000
SUMMARY_ITER = 1
DECAY_STEPS = 10000
DECAY_RATE = 0.5
STAIRCASE = True
SAVE_ITER = 3000

OUTPUT_DIR = 'data/output'