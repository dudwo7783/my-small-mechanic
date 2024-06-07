import os
import io
import re
import collections
import fitz
import camelot
import pdfplumber
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from typing import List

from utils import contains_image_name


class PDFProcessor:
    def __init__(self, NAMESPACE, pdf_file, page_start, page_end, y_range=(39, 575), context_file_root_path = '../pdf_context'):
        
        self.NAMESPACE = NAMESPACE
        
        self.spans = []
        self.contents = []
        self.image_urls = []
        self.csv_ruls = []
        self.content_types = []
        self.block_nums = []
        
        self.pdf_file = pdf_file
        self.doc = fitz.open(pdf_file)
        self.canvas_doc = fitz.open(pdf_file)
        self.plumber_doc = pdfplumber.open(pdf_file)
        self.page_num = page_start
        self.page_end = page_end
        
        self.fitz_page = self.doc.load_page(self.page_num)
        self.fitz_canvas_page = self.canvas_doc.load_page(self.page_num)
        self.plumber_page = self.plumber_doc.pages[self.page_num]
        self.table_processor = TableProcessor(self.pdf_file, self.page_num, self.fitz_page, self.plumber_page)
        self.image_processor = ImageProcessor(self.fitz_page, self.page_num, NAMESPACE, 
                                              context_file_root_path, 
                                              font_path="/Users/yj/Kim/1.work/SKR/8.GenAI/my-small-mechanic/pdf_parser/font/NanumGothicBold.ttf", 
                                              font_size=16)
        
        self.tables = []
        self.table_areas = []
        self.origin_table_areas = []
        
        self.y_range = y_range
        self.context_file_root_path = context_file_root_path
        
        
    def __next__(self):
        
        if self.page_num < self.page_end:
            self.fitz_page = self.doc.load_page(self.page_num)
            self.fitz_canvas_page = self.canvas_doc.load_page(self.page_num)
            self.plumber_page = self.plumber_doc.pages[self.page_num]
            self.table_processor = TableProcessor(self.pdf_file, self.page_num, self.fitz_page, self.plumber_page)
            self.image_processor = ImageProcessor(self.fitz_page, self.page_num, self.NAMESPACE, 
                                              self.context_file_root_path, 
                                              font_path="/Users/yj/Kim/1.work/SKR/8.GenAI/my-small-mechanic/pdf_parser/font/NanumGothicBold.ttf", font_size=16)
            self.page_num += 1
            return self
            
        else:
            raise StopIteration
    
    def set_page_num(self, page_num):
            self.fitz_page = self.doc.load_page(page_num)
            self.fitz_canvas_page = self.canvas_doc.load_page(page_num)
            self.plumber_page = self.plumber_doc.pages[self.page_num]
            self.table_processor = TableProcessor(self.pdf_file, self.page_num, self.fitz_page, self.plumber_page)
            self.image_processor = ImageProcessor(self.fitz_page, page_num, self.NAMESPACE, 
                                              self.context_file_root_path, 
                                              font_path="/Users/yj/Kim/1.work/SKR/8.GenAI/my-small-mechanic/pdf_parser/font/NanumGothicBold.ttf", font_size=16)
        
    
    def parse_pdf(self):
        spans_list = []
        contents_list = []
        image_url_list = []
        csv_url_list = []
        contents_type = []
        block_list = []
        
        
        page_tables = self.table_processor.extract_tables()
        if page_tables.fitz_dfs is not None:
            self.table_areas = self.table_processor.table_areas.copy()
            self.origin_table_areas = self.table_processor.origin_table_areas.copy()
            
            valid_table_list, fail_process_idx = self.table_processor.extract_valid_tables(page_tables)
            
            for fail_idx in fail_process_idx:
                del self.table_areas[fail_idx]
                del self.origin_table_areas[fail_idx]
                
            self.tables = valid_table_list
        
        text_rotation = self.is_text_rotate(self.fitz_page)
        if text_rotation:
            self.fitz_page.set_rotation(90)
            self.fitz_canvas_page.set_rotation(90)
            
        is_two_column = self.analyze_page_layout(self.fitz_page, text_rotation)
        blocks = self.fitz_page.get_text("dict")["blocks"]
        blocks_sorted = self.sort_blocks(blocks, is_two_column, self.fitz_page.rect.width / 2, self.fitz_page.rotation)
        
        prev_overlap_tbl_num=-1
        prev_text = ""
        
        for i, b in enumerate(blocks_sorted):
            overlapping=False
            block_num = f"{self.page_num}_{i}"
            bbox = b['bbox']  # Get the bounding box of the block
            caption = str(i + 1)
            
            if b['type'] == 0:  # Block contains text
                # 테이블 bbox안의 텍스트 block이라면 PASS
                block_area = b['bbox']  # 텍스트 블록의 영역 (x0, y0, x1, y1)
                for tbl_num, table_area in enumerate(self.table_areas):
                    if self.is_overlapping(block_area, table_area):
                        overlapping=True
                        if tbl_num != prev_overlap_tbl_num:
                            spans_list.append(None)
                            tbl_df, content = self.table_processor.post_processing_table(self.tables[tbl_num], text_rotation)
                            
                            csv_url_root_dir = self.context_file_root_path
                            tbl_csv_file_path = os.path.join('csv', self.NAMESPACE, f'{self.page_num}_{tbl_num}_table.csv')
                            tbl_csv_url = os.path.join(csv_url_root_dir, tbl_csv_file_path)
                            tbl_df.to_csv(tbl_csv_url, index=False)
                            
                            csv_url_list.append(tbl_csv_file_path)
                            contents_list.append(content)
                            # Save Table Image
                            rect = fitz.Rect(self.origin_table_areas[tbl_num].bbox)
                            tab_bbox = self.table_processor.get_rotated_bbox(self.fitz_page, rect)
        
                            img_url = self.image_processor.save_table(tbl_num, tab_bbox, 99, 99, 
                                'jpeg', dpi=300, text_rotation=text_rotation)
                    
                            image_url_list.append(img_url)
                            block_list.append(block_num)
                            contents_type.append('table')
                            prev_overlap_tbl_num = tbl_num
                            break
                        else:
                            # print("테이블 추출 이미 완료")
                            break
                if overlapping:
                    continue
                
                if 'lines' in b:
                    for line in b['lines']:
                        line_text = ''.join(list(map(lambda x: x.get('text', ''), line['spans'])))
                    
                        line_spans = str(set(list(map(lambda x: str((x.get('font'), x.get('color'), int(x.get('size')))), 
                                                    line['spans'])))).replace('{', '').replace('}', '')                    

                        spans_list.append(line_spans)
                        contents_list.append(line_text)
                        csv_url_list.append(None)
                        image_url_list.append(None)
                        contents_type.append('text')
                        block_list.append(block_num)
                
                    
                # Insert text on the pixmap
                # Reading order number as caption
                rect = fitz.Rect(bbox)  # Create a rectangle
                
                self.fitz_canvas_page.insert_text(  # insert footer 50 points above page bottom
                        (bbox[0], bbox[1]), caption, color=(1,0,0)
                    )

                #이미지 캡션을 위해 한줄짜리 block인지 확인 후 저장
                text_lines = [span['text'] for line in b["lines"] for span in line["spans"]]
                text_lines = list(filter(lambda x: (x != '̰') and (x != ' ') and (x != '•'),text_lines))
                if len(text_lines) == 1:
                    
                    spans = b["lines"][0]["spans"][0]
                    if not ((spans['size']==6) and (spans['font']=='HyundaiSansTextKR') and 
                            (spans['color']==16777215)):
                        prev_text = text_lines[0]
                        
            if b['type'] != 0:  # Block contains text
                block_area = b['bbox']
                for tbl_num, table_area in enumerate(self.table_areas):
                    if self.is_overlapping(block_area, table_area):
                        overlapping=True
                        break
                if overlapping:
                    continue
                else:
                    # Save Images
                    xres = b['xres']  # PyMuPDF의 기본 해상도는 72 DPI입니다.
                    yres = b['yres']
                    ext = b['ext']
                    width = b['width']
                    height = b['height']
                    
                    img_url = self.image_processor.save_image(i, prev_text, bbox, xres, yres, width, height, ext, size_ratio=1.3, dpi=300)
                    spans_list.append(None)
                    contents_list.append(None)
                    csv_url_list.append(None)
                    image_url_list.append(img_url)
                    contents_type.append('image')
                    block_list.append(block_num)
                    prev_text = ""  # 이전 텍스트 초기화
                    
        return {'span': spans_list, 'contents': contents_list, 'img_urls': image_url_list, 'csv_urls': csv_url_list,'contents_type': contents_type, 'block_num': block_list}
                    
        
    def is_overlapping(self, area1, area2):
        """
        두 영역이 겹치는지 확인하는 함수.
        area1, area2: (x0, y0, x1, y1) 형식의 영역
        """
        x0_1, y0_1, x1_1, y1_1 = area1
        x0_2, y0_2, x1_2, y1_2 = area2
        return not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)
        
            
    def sort_blocks(self, blocks, is_two_column, page_mid, page_rotation=False):
        blocks = [
            b for b in blocks 
            if b['bbox'][1] >= self.y_range[0] and b['bbox'][3] <= self.y_range[1]
        ]

        if is_two_column:
            left_blocks = [b for b in blocks if b['bbox'][0] < page_mid]
            right_blocks = [b for b in blocks if b['bbox'][0] >= page_mid]
            sorted_blocks = sorted(left_blocks, key=lambda b: (b['bbox'][1], b['bbox'][0])) + \
                            sorted(right_blocks, key=lambda b: (b['bbox'][1], b['bbox'][0]))
        else:
            if page_rotation:
                sorted_blocks = sorted(blocks, key=lambda b: (b['bbox'][0], b['bbox'][1]))
            else:
                sorted_blocks = sorted(blocks, key=lambda b: (b['bbox'][1], b['bbox'][0]))
        return sorted_blocks
        
    def analyze_page_layout(self, page, text_rotate):
        tbls = page.find_tables().tables
        is_two_column = True
        blocks = page.get_text("dict")["blocks"]

        # 텍스트가 로테이트 되어있거나,
        if text_rotate:
            is_two_column = False
        # 테이블이 있는데 width가 페이지 넓이 절반보다 크거나,    
        tbls = page.find_tables().tables
        if tbls:
            for tbl in tbls:
                bbox = tbl.bbox
                if (bbox[2] - bbox[0]) > page.rect.width/2:
                    is_two_column = False
        # Block 있는데 width가 페이지 넓이 절반보다 크거나.
        if len(list(filter(lambda x: (x['bbox'][2] - x['bbox'][0])>(page.rect.width/2), blocks)))>0:
            is_two_column = False
        return is_two_column
    
    
    @staticmethod
    def is_text_rotate(fitz_page):
        blocks = fitz_page.get_text("dict")['blocks']
        if fitz_page.find_tables().tables:
            for b in blocks:
                if 'lines' in b:
                    for line in b["lines"]:
                        if 'dir' in line:
                            if line['dir'] == (0.0, -1.0):
                                return True
        else:
            return False
        return False
    
        
class PageTables:
    def __init__(self, fitz_dfs:List[pd.DataFrame]=[], plumber_dfs:List[pd.DataFrame]=[], 
                 camelot_dfs:List[pd.DataFrame]=[], span_camelot_dfs:List[pd.DataFrame]=[]):
        self.fitz_dfs = fitz_dfs
        self.plumber_dfs = plumber_dfs
        self.camelot_dfs = camelot_dfs
        self.span_camelot_dfs = span_camelot_dfs
        
        self.fitz_dfs_count = len(fitz_dfs)
        self.plumber_dfs_count = len(plumber_dfs)
        self.camelot_dfs_count = len(camelot_dfs)
            
class TableProcessor:
    
    def __init__(self, pdf_file, page_num, fitz_page, plumber_page):
        self.pdf_file = pdf_file
        self.page_num = page_num
        self.fitz_page = fitz_page
        self.plumber_page = plumber_page
        self.fitz_tbl_list, self.table_areas, self.origin_table_areas = self.extract_fitz_tables()
        

        
    def extract_fitz_tables(self):
        fitz_tbl_list = []
        table_areas = []
        origin_table_areas = []
        for fitz_tbl in self.fitz_page.find_tables().tables:
            fitz_df = fitz_tbl.to_pandas()
            if (fitz_tbl.row_count == 0) or (fitz_df.shape[0] == 0):
                continue
            else:
                # 테이블 길이 제한으로 테이블이 회전되어있다면 DataFrame 회전
                if PDFProcessor.is_text_rotate(self.fitz_page):
                    fitz_df = self.rotate_dataframe(fitz_df)
                fitz_tbl_list.append(fitz_df)
                table_areas.append(self.get_rotated_bbox(self.fitz_page, fitz.Rect(fitz_tbl.bbox)))
                
        if PDFProcessor.is_text_rotate(self.fitz_page):
            self.fitz_page.set_rotation(90)
            
        for fitz_tbl in self.fitz_page.find_tables().tables:
            fitz_df = fitz_tbl.to_pandas()
            if (fitz_tbl.row_count == 0) or (fitz_df.shape[0] == 0):
                continue
            else:
                origin_table_areas.append(fitz_tbl)  
                
        if PDFProcessor.is_text_rotate(self.fitz_page):
            self.fitz_page.set_rotation(0)
            
        if len(fitz_tbl_list) == 0:
            return [], [], []
        else:    
            return fitz_tbl_list, table_areas, origin_table_areas
        
    def rotate_dataframe(self, _df):
        df = _df.copy()
        
        # 컬럼 정제 및 Empty 컬럼 제거
        df.columns = [re.sub(r'^\d+-', '', col) for col in df.columns]
        null_sum = df.isnull().sum()
        null_cols = null_sum[null_sum == df.shape[0]].index
        df = df.drop(null_cols, axis=1)

        # 컬럼을 첫번쨰 row로 
        first_col_as_df = pd.DataFrame([df.columns.tolist(), df.iloc[0,:].values], columns=range(0, len(df.columns)))

        empty_str_cols = np.where(first_col_as_df.iloc[1, :].values == '')[0].tolist()
        first_col_as_df.loc[0, empty_str_cols] = ''

        None_str_cols = np.where(first_col_as_df.iloc[1, :].values is None)[0].tolist()
        first_col_as_df.loc[0, None_str_cols] = None
        first_col_as_df = first_col_as_df.loc[[0],:]

        df.columns = range(0, len(df.columns))
        df = pd.concat([first_col_as_df, df])
        df = df.set_index(df.columns[0])
        df = df.T
        df = df.iloc[:,list(range(len(df.columns)-1,-1, -1))]
        df.columns = list(map(lambda x: 'Col' if x is None else x,df.columns.tolist()))
        df = df.reset_index(drop=True)
        
        return df
    
    def get_rotated_bbox(self, page, rect):
        """
        Adjust the bounding box coordinates for a page that is rotated 90 degrees clockwise.
        
        Args:
        - page: The page object from PyMuPDF.
        - rect: The original bounding box as a fitz.Rect object.
        
        Returns:
        - A fitz.Rect object representing the adjusted bounding box.
        """
        if page.rotation == 90:
            page_width = page.rect.width
            # page_height = page.rect.height
            
            # Transform the rectangle coordinates
            new_x0 = rect.y0
            new_y0 = page_width - rect.x1
            new_x1 = rect.y1
            new_y1 = page_width - rect.x0
            
            # Create a new rectangle with the adjusted coordinates
            adjusted_rect = fitz.Rect(new_x0, new_y0, new_x1, new_y1)
            return adjusted_rect
        
        # If no rotation or a different rotation, return the original rectangle
        return rect
        
    def extract_tables(self):
        camelot_tbl_list = []
        camelot_span_tbls_list = []  # Span된 테이블의 fill 한 결과 테이블
        plumber_tbl_list = []
        
        # Fitz 테이블이 없을시 Break
        if len(self.fitz_tbl_list) == 0:
            return PageTables()
            
        '''
        Camelot Table 추출
        '''
        camelot_tbls = camelot.read_pdf(self.pdf_file, 
                                        pages=f"{self.page_num+1}",
                                        flavor='lattice', 
                                        shift_text=['l', 't'], 
                                        strip_text=' .\n')
        camelot_span_tbls = camelot.read_pdf(self.pdf_file, 
                                             pages=f"{self.page_num+1}",
                                             flavor='lattice', 
                                             shift_text=['l', 't'], 
                                             copy_text=['v','h'], 
                                             strip_text=' .\n')
        
        for camelot_tbl, camelot_span_tbl in zip(camelot_tbls, camelot_span_tbls):
            # camelot 테이블이 비어있다면 PASS
            if (camelot_tbl.df == '').all().all() or camelot_tbl.df.isna().all().all():
                continue
            else:
                camelot_tbl_list.append(camelot_tbl.df)
                camelot_span_tbls_list.append(camelot_span_tbl.df)
                
        '''
        Plumber 테이블 추출
        '''
        plumber_tbls = self.plumber_page.extract_tables()
 
        for plumber_tbl in plumber_tbls:
            plumber_tbl_df = pd.DataFrame(plumber_tbl)
            # Plumber 테이블에 cid라는 값이 존재한다면 PASS
            if plumber_tbl_df.astype(str).apply(lambda x: x.str.contains('cid')).any().any():
                continue
            else:
                plumber_tbl_list.append(plumber_tbl_df)
                
        page_tables = PageTables(self.fitz_tbl_list, plumber_tbl_list, camelot_tbl_list, camelot_span_tbls_list)

        return page_tables
    
    
    def find_end_columns_index(self, _fitz_df, _camelot_df, plumber_df):
        # retry=True
        fitz_df = _fitz_df.copy()
        camelot_df = _camelot_df.copy()
        
        try_count = 0
        while try_count<2:        
            fitz_columns = list(map(lambda x: '' if 'Col' in x else x, fitz_df.columns))
            
            if plumber_df.shape != (0,0):
                plumber_columns = list(map(lambda x: '' if x is None else x, plumber_df.loc[0].fillna('').values))
                plumber_re_row = list(map(lambda x: set(re.sub(r'[^\w\s]', '', x).replace('\n','').replace(' ','')), plumber_columns))
            else:
                plumber_columns = []
                plumber_re_row = []
            if _camelot_df.shape != (0,0):
                camelot_first_row = list(map(lambda x: '' if x is None else x, camelot_df.loc[0].fillna('').values))
                camelot_re_row = list(map(lambda x: set(re.sub(r'[^\w\s]', '', x).replace('\n','').replace(' ','')), camelot_first_row))
            else:
                camelot_first_row = []
                camelot_re_row = []
            
            if len(fitz_columns) == len(camelot_first_row):
                # fitz가 plubmer와 같다면 그대로 사용
                if fitz_columns == plumber_columns:
                    pass
                # fitz가 plumber가 컬럼 개수는 같은데 값 다른 경우(fitz의 이상한 문자가 들어간 경우)
                # 문자를 찾아서 fitz의 컬럼을 다시 바꾸고 이상한 값을 제거
                elif (len(fitz_columns) == len(plumber_columns)) and (fitz_columns != plumber_columns):
                    plumber_match = False
                    for col_start_row_idx, row in fitz_df.iterrows():

                        fitz_re_row = list(map(lambda x: set(re.sub(r'[^\w\s]', '', x).replace('\n','').replace(' ','')), 
                                            row.fillna('').values))

                        if fitz_re_row == plumber_re_row:
                            plumber_match = True
                            break
                    if plumber_match:
                        fitz_df.columns = plumber_columns
                        fitz_df = fitz_df.loc[col_start_row_idx+1:]
                else:
                    pass
                        
                camelot_match = False
                for col_end_row_idx, row in fitz_df.iterrows():
                    fitz_re_row = list(map(lambda x: set(re.sub(r'[^\w\s]', '', x).replace('\n','').replace(' ','')), 
                                        row.fillna('').values))

                    if fitz_re_row == camelot_re_row:
                        camelot_match = True
                        break
                    
                columns_list = [fitz_df.columns.tolist()]
                if camelot_match:
                    columns_list.extend(fitz_df.loc[:col_end_row_idx-1].fillna('').values.tolist())
                
            # fitz와 camelot의 길이는 달랐지만, camelot과 plumber의 길이가 같은 경우
            elif len(plumber_columns) == len(camelot_first_row):
                
                
                plubmer_none_columns = list(filter(lambda x: x == '', plumber_columns))
                plumber_columns = list(map(lambda x: 'Col' if x == '' else x, plumber_columns))
                columns_list = [plumber_columns]
                if len(plubmer_none_columns) > 0:
                    plumber_next_columns = plumber_df.loc[1].fillna('').values.tolist()
                    plumber_columns = list(map(lambda x: 'Col' if x == '' else x, plumber_next_columns))
                    columns_list.append(plumber_next_columns)
            # camelot이 전부 다른데, fitz랑 plumber는 맞는 경우
            elif (len(camelot_first_row) != len(fitz_columns)) & (len(camelot_first_row) != len(plumber_columns)) & (len(fitz_columns) == len(plumber_columns)):
                plumber_match = False
                for col_start_row_idx, row in fitz_df.iterrows():

                    fitz_re_row = list(map(lambda x: set(re.sub(r'[^\w\s]', '', x).replace('\n','').replace(' ','')), 
                                        row.fillna('').values))

                    if fitz_re_row == plumber_re_row:
                        plumber_match = True
                        break
                if plumber_match:
                    fitz_df.columns = plumber_columns
                    fitz_df = fitz_df.loc[col_start_row_idx+1:]
                columns_list = [fitz_df.columns.tolist()]
            else:
                if (try_count==0) and ('Col' in fitz_df.columns[0]):
                    print('retry')
                    fitz_df.iloc[:,1] = fitz_df.iloc[:,0].fillna('').astype(str) + '\n' + fitz_df.iloc[:,1].fillna('').astype(str)
                    fitz_df_columns = fitz_df.columns.tolist()
                    fitz_df_columns[1] = fitz_df_columns[0] + '_' + fitz_df_columns[1]
                    fitz_df.columns = fitz_df_columns
                    fitz_df = fitz_df.iloc[:, 1:]
                    try_count = try_count+1
                    continue
                else:
                    return -1, None
                
                
            columns_list = pd.DataFrame(columns_list).drop_duplicates().values.tolist()
                
            return columns_list, fitz_df
        
        
    def replace_col_and_merge(self, data, isomorphic=True):
        num_rows = len(data)
        num_cols = len(data[0])
        
        # 'Col'이 포함된 컬럼 이름을 이전 원소의 값으로 대체
        # [['짧게 누를 때\n(0.8초 미만)', '길게 누를 때\n(0.8초 이상)', 'Col']]
        for row in range(num_rows):
            prev_value = ''
            for col in range(num_cols):
                if 'Col' in data[row][col]:
                    if prev_value == '':
                        data[row][col] = data[row][col+1]
                        data[row][col+1] = 'Col'
                        prev_value = data[row][col]
                    elif isomorphic:
                        data[row][col] = prev_value + f"_{row}"
                    else:
                        data[row][col] = prev_value
                else:
                    prev_value = data[row][col]
        
        result = []
        for col in range(num_cols):
            column_str = ""
            for row in range(num_rows):
                if data[row][col] != '':
                    column_str += data[row][col] + "_"
                else:
                    column_str += data[row][col]
            result.append(column_str.rstrip('_'))
        
        result = list(map(lambda x: x.replace('\n', ''), result))
        return result

    def reduce_columns(self, lst, target_count):
        reduce_list = lst.copy()
        
        while len(reduce_list) > target_count:
            lst_counter = collections.Counter(reduce_list)
            most_key = lst_counter.most_common(1)[0][0]
            reduce_list.remove(most_key)
        
        return reduce_list    
    
    def extract_valid_tables(self, page_tables:PageTables):
        process_table_list = []
        fail_idx = []
        
        fitz_size = page_tables.fitz_dfs_count
        fitz_dfs = page_tables.fitz_dfs
        camelot_size = page_tables.camelot_dfs_count
        camelot_dfs = page_tables.camelot_dfs
        plumber_size = page_tables.plumber_dfs_count
        plumber_dfs = page_tables.plumber_dfs
        span_camelot_dfs = page_tables.span_camelot_dfs
        
            
        if fitz_size == camelot_size:
            for i in range(fitz_size):
                isomorphic = fitz_dfs[i].shape == camelot_dfs[i].shape
                
                if plumber_size == 0:
                    plumber_dfs.append(pd.DataFrame())
                    
                columns_str_list, _ = self.find_end_columns_index(fitz_dfs[i], camelot_dfs[i],plumber_dfs[i])
                if columns_str_list == -1:
                    fail_idx.append(i)
                    continue
                
                process_column = self.replace_col_and_merge(columns_str_list, isomorphic=isomorphic)
                process_column = self.reduce_columns(process_column, camelot_dfs[i].shape[1])
                cur_df = span_camelot_dfs[i].copy()
                if len(cur_df.columns) == len(process_column):
                    cur_df.columns = process_column
                # 만약에 길이가 맞지 않다면 plumber에서 첫번째 row 중에 Null인 컬럼을 제외하고 나머지에 매핑
                elif len(np.where(~(cur_df.loc[0]==''))[0]) == len(plumber_dfs[i].columns):
                    process_column = ['Col'] * len(cur_df.columns)
                    for camelot_col_idx, plumber_col in zip(np.where(~(cur_df.loc[0]==''))[0], plumber_dfs[i].loc[0].values):
                        process_column[camelot_col_idx] = plumber_col
                    process_column = self.replace_col_and_merge([process_column], isomorphic=True)
                    cur_df.columns = process_column
                else:
                    fail_idx.append(i)
                    continue
                
                # process_table_list[i] = cur_df
                process_table_list.append(cur_df)
                
        else:
            if fitz_size == plumber_size:
                for i in range(fitz_size):
                    columns_str_list, cur_df = self.find_end_columns_index(fitz_dfs[i], pd.DataFrame(),plumber_dfs[i])
                    if columns_str_list == -1:
                        fail_idx.append(i)
                        continue
                    cur_df = cur_df.fillna(method='ffill')
                    process_column = self.replace_col_and_merge(columns_str_list, isomorphic=True)
                    cur_df.columns = process_column
                    
                    # process_table_list[i] = cur_df
                    process_table_list.append(cur_df)
            elif fitz_size != plumber_size:
                for i in range(0,fitz_size):
                    columns_str_list = [fitz_dfs[i].columns.tolist()]
                    if columns_str_list == -1:
                        fail_idx.append(i)
                        continue
                    process_column = self.replace_col_and_merge(columns_str_list, isomorphic=True)
                    cur_df = fitz_dfs[i].copy().fillna(method='ffill')
                    cur_df.columns = process_column
                    # process_table_list[i] = cur_df
                    process_table_list.append(cur_df)
            else:
                return [], []
            
        return process_table_list, fail_idx

    def post_processing_table(self, df, text_rotation=False):
        df.columns = df.columns = [re.sub(r'^\d+-', '', col) for col in df.columns]
        null_sum = df.isnull().sum()
        null_cols = null_sum[null_sum == df.shape[0]].index
        df = df.drop(null_cols, axis=1)
        if text_rotation:
            
            first_col_as_df = pd.DataFrame([df.columns.tolist(), df.iloc[0,:].values], columns=range(0, len(df.columns)))
            
            empty_str_cols = np.where(first_col_as_df.iloc[1, :].values == '')[0].tolist()
            first_col_as_df.loc[0, empty_str_cols] = ''
            
            None_str_cols = np.where(first_col_as_df.iloc[1, :].values is None)[0].tolist()
            first_col_as_df.loc[0, None_str_cols] = None
            first_col_as_df = first_col_as_df.loc[[0],:]
            
            df.columns = range(0, len(df.columns))
            df = pd.concat([first_col_as_df, df]).fillna(method='ffill')
            df = df.set_index(df.columns[0])
            df = df.T
            df = df.iloc[:,list(range(len(df.columns)-1,-1, -1))]

        df_cols = df.columns.tolist()
        for i in range(1, len(df_cols)):
            if "Col" in df_cols[i]:
                df_cols[i] = df_cols[i-1]  # "COl"을 이전 인덱스의 값으로 대체합니다.
                
        df.columns = df_cols
        df = df.fillna('')
        img_index = df.applymap(lambda x: contains_image_name(x))
        
        df[img_index] = '이미지 참조'
        
        df_md = df.to_markdown()
        
        
        return df, df_md    
    
class ImageProcessor:
    
    def __init__(self, fitz_page, page_num, NAMESPACE, context_file_root_path, font_path="./font/NanumGothicBold.ttf", font_size=16):
        self.fitz_page = fitz_page
        self.page_num = page_num
        self.font_path = font_path
        self.font_size = font_size
        self.NAMESPACE = NAMESPACE
        self.context_file_root_path = context_file_root_path
    
    def save_table(self, index, bbox, xres, yres, ext, dpi=300, text_rotation=False):
        
        text_rotation = PDFProcessor.is_text_rotate(self.fitz_page)
        if text_rotation:
            self.fitz_page.set_rotation(0)
        x_scale = dpi/2 /xres
        y_scale = dpi/2 / yres
        matrix = fitz.Matrix(x_scale, y_scale)  # 변환 행렬 생성
        
        pix = self.fitz_page.get_pixmap(matrix=matrix, clip=bbox)# 선택한 bbox 영역의 이미지를 추출 (해상도 조절)
        image = Image.open(io.BytesIO(pix.tobytes()))# PIL 이미지로 변환
        
        if text_rotation:
            image = image.rotate(270,expand=True)
            self.fitz_page.set_rotation(90)
            
        img_root_dir = self.context_file_root_path
        img_file_path = f"image/{self.NAMESPACE}/f'{self.page_num}_{index}_table.{ext}'"
        img_url = os.path.join(img_root_dir, img_file_path)
        image.save(img_url)
        return img_file_path
    
    def save_image(self, index, prev_text, bbox, xres, yres, width, height, ext, size_ratio=1.3, dpi=300):
        
        x_scale = dpi /xres  # PyMuPDF의 기본 해상도는 72 DPI입니다.
        y_scale = dpi / yres
        matrix = fitz.Matrix(x_scale, y_scale)  # 변환 행렬 생성
        
        # 선택한 bbox 영역의 이미지를 추출 (해상도 조절 적용)
        pix = self.fitz_page.get_pixmap(matrix=matrix, clip=bbox)
        
        # PIL 이미지로 변환
        image = Image.open(io.BytesIO(pix.tobytes()))
        image = image.resize((int(width*size_ratio), int(height*size_ratio)), Image.Resampling.LANCZOS)
        
        # 이미지에 텍스트 추가
        draw = ImageDraw.Draw(image)
        # 폰트는 시스템에 따라 경로가 다를 수 있으며, 필요에 따라 수정해야 할 수 있습니다.
        font = ImageFont.truetype(self.font_path, self.font_size)  # 폰트와 크기 설정
        draw.text((10, 10), prev_text, fill='black', font=font)
        
        
        img_root_dir = self.context_file_root_path
        img_file_path = f"image/{self.NAMESPACE}/f'{self.page_num}_{index}_img.{ext}'"
        img_url = os.path.join(img_root_dir, img_file_path)
        image.save(img_url)
        return img_file_path


        

        