from pathlib import Path
from sqlalchemy import text
import sys


sys.path.append(str(Path(__file__).parent.parent))
from models.models import Image, Base, engine, session

class ImageController:

    def __init__(self):
        self.__session = session

    def get_image(self, image_id):
        return self.__session.query(Image).filter(Image.img_id == image_id).first()

    def insert_image(self, image):
        self.__session.add(image)
        self.__session.commit()

    def update_image(self, image):
        updated_image = self.__session.query(Image).filter(Image.img_id == image.img_id).first()
        updated_image.img_path = image.img_path
        updated_image.img_class = image.img_class
        updated_image.img_concept_id = image.img_concept_id
        updated_image.img_things_concept_id = image.img_things_concept_id
        self.__session.commit()

    def delete_image(self, image_id):
        deleted_image = self.__session.query(Image).filter(Image.img_id == image_id).first()
        if deleted_image is not None:
            self.__session.delete(deleted_image)
            self.__session.commit()
    
    def run_query(self, query):
        if isinstance(query, str):
            query = text(query)
        result = self.__session.execute(query)
        self.__session.commit()
        return result

if __name__ == '__main__':
    image_controller = ImageController()
    image = Image(img_path='path2', img_class='class', img_concept='concept', img_concept_id=1, img_things_concept_id=1, img_condition=1, split='training')
    #image_controller.insert_image(image)
    #cursor = image_controller.run_query('DELETE FROM image')
    
