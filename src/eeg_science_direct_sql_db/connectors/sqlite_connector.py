import sqlalchemy
import os
from pathlib import Path
#import pyodbc

class SQLiteConnector:

    def __init__(self):
        pass

    def connect_sql_alchemy(conn_str):
        db_path = conn_str.split('////')[-1]
        if not os.path.exists(db_path):
            with open(db_path, 'w') as f:
                f.write('')
        conn_str = 'sqlite:///' + str(Path(__file__).parent.parent.parent.parent) + '/' + db_path
        return sqlalchemy.create_engine(conn_str).connect()
    
    #def connect_pydobc(conn_str):
    #    return pyodbc.connect(conn_str)
    

if __name__ == '__main__':
    from dotenv import load_dotenv, find_dotenv
    import os
    load_dotenv(find_dotenv())
    
    engine = SQLiteConnector.connect_sql_alchemy(os.getenv('SQLITE_DB_SQL_ALCHEMY_CONN_STR'))

    
    
    
