import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.eeg_science_direct_sql_db.controllers.image_controller import ImageController, Image
from src.datasets.ImageSciencedirect import ImageSciencedirect

class ImageDBLoader:

    def __init__(self):
        self.__dataset = ImageSciencedirect()
        self.__db_controller = ImageController()
    
    def load(self):
        self.__db_controller.run_query("DELETE FROM image")
        self.__load_split("test")
        self.__load_split("training")
    
    def __load_split(self, split):
        dataset = self.__dataset.create(split)
        for img_obj in dataset:
            image = Image(
                img_path=img_obj["img_path"],
                img_class=img_obj["class"],
                img_concept=img_obj["concept"],
                img_concept_id=img_obj["concept_id"],
                img_things_concept_id=img_obj["things_concept_id"],
                img_condition=img_obj["img_cond"],
                split=img_obj["split"]
            )
            self.__db_controller.insert_image(image)
        

if __name__ == '__main__':
    image_db_loader = ImageDBLoader()
    image_db_loader.load()
