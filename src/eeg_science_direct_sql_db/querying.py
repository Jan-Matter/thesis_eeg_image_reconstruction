from pathlib import Path
import sys
import os
from sqlalchemy import text
import numpy as np
from PIL import Image
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())



sys.path.append(str(Path(__file__).parent.parent.parent))

from src.eeg_science_direct_sql_db.models.models import session

class EEGScienceDirectSQLDBQueryier:

    def __init__(self):
        pass

    def run_query(self, query):
        query = query.strip()
        if not 'group' in query.lower():
            cursor = session.execute(text(query))
            columns = cursor.keys()
            for row in cursor:
                row_dict =  dict(zip(columns, row))
                if 'img_path' in row_dict:
                    row_dict['img'] = self.__load_image(row_dict['img_path'])
                yield row_dict
            return
        
        if 'avg' in query.lower():
            transform = 'avg'
        if 'median' in query.lower():
            transform = 'median'
        if 'std' in query.lower():
            transform = 'std'
        
        if transform is not None:
            parsed__avg_column = self.__parse_group_by_column(query, transform)
            transform_column = parsed__avg_column['column']
            transform_group_by = parsed__avg_column['group_by']
            begin_char = parsed__avg_column['begin_char']
            end_char = parsed__avg_column['end_char']
            query = query.replace(query[begin_char:end_char], transform_column)
            
        kws = ['group', 'having', 'order', 'limit']
        if 'having' in query or 'order' in query:
            raise NotImplementedError('HAVING and ORDER BY are not supported yet')
        
        limit = None
        if 'limit' in query.lower():
            limit = int(query.lower().split('limit')[1].split()[0])
        kw_in_query = [kw for kw in kws if kw in query.lower()]
        kw_start_indexes = [query.lower().find(kw) for kw in kws if kw in query.lower()]
        group_by_columns_end_index = kw_start_indexes[1] if len(kw_start_indexes) > 1 else len(query)
        group_by_columns = [column.strip()
                            for column in 
                            ' '.join(query[kw_start_indexes[0]:group_by_columns_end_index]\
                                .split()[2:])\
                                .split(',')
                    ]
        query_before_group_by = query[:kw_start_indexes[0]]
        query_for_group_by = query_before_group_by + 'ORDER BY ' + ', '.join(group_by_columns)

        from_index = query_for_group_by.lower().find('from')
        for column in group_by_columns + ['image.img_condition']:
            if column not in query_for_group_by[:from_index]:
                query_for_group_by = query_for_group_by.replace(query[:7], 'SELECT ' + column + ', ')
        
        cursor = session.execute(text(query_for_group_by))
        columns = cursor.keys()
        last_row = {}
        images = {}
        data = []
        data_cache = {}
        row_count = 1
        group_by_columns_suffix = [column.split('.')[-1] for column in group_by_columns]
        for row in cursor:
            next_row = dict(zip(columns, row))
            if 'img_path' in last_row:
                img_path = last_row['img_path']
                if img_path not in images:
                    images[img_path] = self.__load_image(last_row['img_path'])
            if 'data_path' in last_row:
                data_path = os.getenv('SCIENCE_DIRECT_PREPROCESSED_DATA_FOLDER') + '/' + last_row['data_path']
                if data_path not in data_cache:
                    data_loaded = np.load(data_path)
                    data_cache[data_path] = data_loaded
                data_loaded = data_cache[data_path]
                try:
                    data.append(data_loaded[last_row['img_condition'] - 1])
                except:
                    pass

            if len(last_row) > 0 and not self.__equal_group_by_columns(last_row, next_row, group_by_columns_suffix):
                output_row = {}
                for column in group_by_columns_suffix:
                    if column in ['img_path', 'data_path']:
                        continue
                    output_row[column] = last_row[column]
                if len(images) > 0:
                    output_row['imgs'] = list(images.values())
                if transform is not None and len(data) > 0:
                    output_data = np.array(data)
                    output_data = np.nan_to_num(output_data, nan=0)
                    if 'fourier' in query.lower():
                        output_data = np.abs(output_data)
                    count = 0
                    if transform == 'avg':
                        ordered_transform_group_by = sorted(transform_group_by)
                        for axis in ordered_transform_group_by:
                            axis -= count
                            output_data = np.mean(output_data, axis=axis)
                            count += 1
                    if transform == 'median':
                        ordered_transform_group_by = sorted(transform_group_by)
                        for axis in ordered_transform_group_by:
                            axis -= count
                            output_data = np.median(output_data, axis=axis)
                            count += 1
                    if transform == 'std':
                        ordered_transform_group_by = sorted(transform_group_by)
                        for axis in ordered_transform_group_by:
                            axis -= count
                            output_data = np.std(output_data, axis=axis)
                            count += 1
                    output_row['data'] = output_data
                yield output_row
                row_count += 1
                if limit is not None and row_count > limit:
                    return

                images = {}
                data = []
                data_cache = {}
            last_row = next_row

    
    def __parse_group_by_column(self, query, transform):
        lower_query = query.lower()
        begin_char = lower_query.find(transform)
        end_char = lower_query.find(')', begin_char) + 1
        contents = [content.strip()
                    for content in  lower_query\
                        .split(transform)[1]\
                        .split('(')[1]\
                        .split(')')[0]\
                        .split(',')
                    ]
        column = contents[0]
        group_by_idxs = contents[1:]
        return {
            'column': column,
            'group_by': [int(group_by) for group_by in group_by_idxs],
            'begin_char': begin_char,
            'end_char': end_char
        }
    
    def __equal_group_by_columns(self, last_row, next_row, group_by_columns):
        return all([last_row[column] == next_row[column] for column in group_by_columns])
    
    

    def __load_image(self, img_path):
        img =Image.open(img_path).convert('RGB')
        return np.asarray(img, dtype="int32")
            


if __name__ == '__main__':
    query = """
    SELECT 
        image.img_path
    FROM image
    """
    eeg_science_direct_sql_db_queryier = EEGScienceDirectSQLDBQueryier()
    cursor = eeg_science_direct_sql_db_queryier.run_query(query)
    for idx, row in enumerate(cursor):
        print(idx, ' ', row['img'])

    