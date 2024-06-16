from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv, find_dotenv
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
load_dotenv(find_dotenv())

from src.eeg_science_direct_sql_db.connectors.sqlite_connector import SQLiteConnector

Base = declarative_base()
