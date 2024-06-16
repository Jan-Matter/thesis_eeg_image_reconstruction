from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker
from pathlib import Path
import sys
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

sys.path.append(str(Path(__file__).parent.parent))
from db_imports import Base



from src.eeg_science_direct_sql_db.connectors.sqlite_connector import SQLiteConnector

class Image(Base):
    __tablename__ = 'image'
    img_id = Column(Integer, primary_key=True)
    img_path = Column(String)
    img_class = Column(String)
    img_concept = Column(String)
    img_concept_id = Column(Integer)
    img_things_concept_id = Column(Integer)
    img_condition = Column(Integer)
    split = Column(String)

db_path = os.getenv('SQLITE_DB_SQL_ALCHEMY_CONN_STR').split('////')[-1]
if not os.path.exists(db_path):
    with open(db_path, 'w') as f:
        f.write('')
engine = SQLiteConnector.connect_sql_alchemy(os.getenv('SQLITE_DB_SQL_ALCHEMY_CONN_STR'))
session = sessionmaker(bind=engine)()
