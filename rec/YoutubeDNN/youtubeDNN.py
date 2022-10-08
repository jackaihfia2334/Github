import os
import time
import pickle
import random
from datetime import datetime
import collections

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

# 从utils里面导入函数
from utils import gen_data_set, gen_model_input, train_youtube_model
from utils import get_embeddings, get_youtube_recall_res

import warnings
warnings.filterwarnings('ignore')
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 1显示所有信息， 2显示error和warnings, 3显示errors