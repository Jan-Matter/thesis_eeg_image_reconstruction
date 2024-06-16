import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd

load_dotenv(find_dotenv())

class ImageSciencedirect:

    def __init__(self, **kwargs):
        self.__data_path = str(Path(__file__).parent.parent.parent / os.getenv("SCIENCE_DIRECT_DATA_FOLDER"))
        self.__metadata = self.__load_metadata()
        self.__concept_to_class_mapping = self.__load_concept_to_class_mapping()
    
    @property
    def metadata(self):
        return self.__metadata
    
    @property
    def concept_to_class_mapping(self):
        return self.__concept_to_class_mapping


    def create(self, split):
        if split == "training":
            directory = self.__data_path + f"/train_images"
        else:
            directory = self.__data_path + f"/test_images"
        img_dirs = os.listdir(directory)
        img_data = []
        if split == "training":
            things_concept_mapping = self.metadata["train_img_concepts_THINGS"]
        else:
            things_concept_mapping = self.metadata["test_img_concepts_THINGS"]
        for img_dir in img_dirs:
            img_dir_path = 'data' + directory.split('data')[-1] + f"/{img_dir}"
            img_files = os.listdir(img_dir_path)
            for img_file in img_files:
                if split == "training":
                    img_cond = (int(img_dir.split("_")[0]) - 1) * 10 + int(img_file.split("_")[-1].split(".")[0][:-1])
                else:
                    img_cond = int(img_dir.split("_")[0])
                img_path = img_dir_path + f"/{img_file}"
                things_concept_id = things_concept_mapping[img_cond - 1].split("_")[0]
                img_class = self.concept_to_class_mapping[int(things_concept_id)]\
                    if int(things_concept_id) in self.concept_to_class_mapping\
                        else "unknown"
                img_obj =  {
                    "concept": img_dir.split("_")[-1],
                    "class": img_class,
                    "img_cond": int(img_cond),
                    "concept_id": int(img_dir.split("_")[0]),
                    "things_concept_id": int(things_concept_id),
                    "image_id": img_file.split("_")[1].split(".")[0],
                    "img_path": img_path,
                    "split": split,
                }
                img_data.append(img_obj)

        return img_data
    
    def __load_metadata(self):
        path = self.__data_path + "/image_metadata.npy"
        return np.load(path, allow_pickle=True).item()

    def __load_concept_to_class_mapping(self):
        path = self.__data_path + "/category_mat_manual.tsv"
        df = pd.read_csv(path, sep="\t")
        mapping = {}
        for index, row in df.iterrows():
            for column in df.columns:
                if row[column] == 1:
                    mapping[index + 1] = column
        return mapping

if __name__ == "__main__":
    img_sciencedirect = ImageSciencedirect()
    img_data = img_sciencedirect.create("training")
    print(img_data[0])
