import sys
sys.path.append('/Users/aleksejkitajskij/Desktop/DataSientist/Class_Learning/')
from class_claster import claster_check

########################################################################################

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive/')
from model_choose import regression_choose

########################################################################################

!git clone https://github.com/Familenko/Model_choose_classification.git
from Model_choose_classification.classification_choose import *