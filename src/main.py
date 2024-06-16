import os
import sys
import numpy as np
from dotenv import load_dotenv, find_dotenv
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

load_dotenv(find_dotenv())

data_path = os.getenv("SCIENCE_DIRECT_PREPROCESSED_DATA_FOLDER_TRAINING_DATASET")
file_names = os.listdir(data_path)
for i, file_path in enumerate(file_names):
    file_path = os.path.join(data_path, file_path)
    data = np.load(file_path)
    print(data)
