
from utils import *
import numpy as np
import pandas as pd

def remove_invalid_row_by_string(_df):
    
    df = _df.copy()
    df['alphanumeric'] = True
    df['korean'] = False
    df['numeric'] = False
    df['image_name'] = False

    df.loc[df['ctype']=='text', 'alphanumeric'] = df.loc[df['ctype']=='text', 'contents'].apply(lambda x: contains_alpha_numeric(x))
    df.loc[df['ctype']=='text', 'korean'] = df.loc[df['ctype']=='text', 'contents'].apply(lambda x: contains_korean(x))
    df.loc[df['ctype']=='text', 'numeric'] = df.loc[df['ctype']=='text', 'contents'].apply(lambda x: contains_numeric(x))
    df.loc[df['ctype']=='text', 'image_name'] = df.loc[df['ctype']=='text', 'contents'].apply(lambda x: contains_image_name(x))
    
    df = df[(df['alphanumeric']) | (df['ctype']=='image')]
    df = df[~((df['ctype']=='text') & (df['contents'].str.contains(r'\.\.\.')))]
    df = df[~((df['ctype']=='text') & (df['numeric']))]
    df = df[~((df['ctype']=='text') & (df['image_name']))]
    
    df['contents'] = df['contents'].str.replace('\u2022', '') # 불릿
    df['contents'] = df['contents'].str.replace('\u2013', '') # 불릿


    return df

def remove_invalid_row_by_span(_df, company='hyundai'):
    df = _df.copy()
    if company == 'hundai':
        df = df[~(df['span']=='"(\'HyundaiSansHeadKR\', 0, 14)"')]  # 마, 카, 타
        df = df[~(df['span']=='"(\'HyundaiSansHeadKR\', 13416, 25)"')]  # 색인
        df = df[~(df['span']=='"(\'HyundaiSansTextKR\', 16777215, 6)"')]  # 불필요 영어
    elif company == 'tesla':
        df = df[~(df['span']=='"(\'NotoSansKR-Regular\', 127, 9)"')]  # 공백
    
    return df

def add_header(_df, company='hyundai'):
    df = _df.copy()
    df['index'] = None
    if company == 'hundai':
        h1_span = '"(\'HyundaiSansHeadKR\', 13416, 28)"'
        h2_span = '"(\'HyundaiSansTextKRMedium\', 13416, 14)"'
        h3_span = '"(\'HyundaiSansTextKRMedium\', 13416, 12)"'
        h4_span = '"(\'HyundaiSansTextKRMedium\', 0, 10)"'
        
        df.loc[df['span']==h1_span, 'index'] = 'h1'
        df.loc[df['span']==h2_span, 'index'] = 'h2'
        df.loc[df['span']==h3_span, 'index'] = 'h3'
        df.loc[df['span']==h4_span, 'index'] = 'h4'
        
    elif company == 'tesla':
        h3_span = '"(\'NotoSansKR-Bold\', 0, 14)"'
        h4_span = '"(\'NotoSansKR-Bold\', 0, 12)"'
        
        df.loc[df['span']==h3_span, 'index'] = 'h3'
        df.loc[df['span']==h4_span, 'index'] = 'h4'
    return df

    
def group_by_block(_df):
    df = _df.copy()
    
    dup_line = df.loc[df['index']=='h4'].groupby(['block_num', 'index']).size()[df.loc[df['index']=='h4'].groupby(['block_num', 'index']).size()>1].reset_index()
    new_h4_line = df.loc[df['index']=='h4'].groupby(['block_num', 'index'])['contents'].apply(lambda x: ''.join(x) + '\n').reset_index().rename(columns={'contents':'contents2'})
    df = df.merge(new_h4_line, on=['block_num','index'], how='left')
    df.loc[~df['contents2'].isnull(), 'contents'] = df.loc[~df['contents2'].isnull(), 'contents2']
    df = df.merge(dup_line, on=['block_num','index'], how='left')
    df = df.drop(index=list(set(df[~df[0].isnull()].index) - set(df[~df[0].isnull()].drop_duplicates().index)))
    df = df.drop(columns=['contents2', 0]).reset_index(drop=True)
    
    # PDF 내 같은 Block 이었던 텍스트끼리 Line 병합
    merge_contents = pd.DataFrame(df[~df['contents'].isnull()].groupby(['block_num'])['contents'].apply(lambda x: ''.join(x))).reset_index()
    merge_contents.columns = ['block_num', 'merge_contents']
    
    # 기존 Line에 Merge Text 후 중복 제거
    df = df.merge(merge_contents, on=['block_num'], how='left')
    m_df = df.drop_duplicates('block_num', keep='first').reset_index(drop=True).drop('contents', axis=1)
    
    
    # Document Grouping
    m_df['doc_group'] = (~m_df['index'].isnull()).cumsum()
    m_df['h1'] = None
    m_df['h2'] = None
    m_df['h3'] = None
    
    m_df.loc[m_df['index']=='h1', 'h1'] = m_df.loc[m_df['index']=='h1', 'merge_contents']
    m_df.loc[m_df['index']=='h2', 'h2'] = m_df.loc[m_df['index']=='h2', 'merge_contents']
    m_df.loc[m_df['index']=='h3', 'h3'] = m_df.loc[m_df['index']=='h3', 'merge_contents']
    
    # h1이 있는데 h2가 없으면 h2는 empty: h1 바로 밑에 text가 있는 경우
    # h2가 있는데 h3가 없으면 h3는 empty: h2 바로 밑에 text가 있는 경우
    m_df.loc[(~m_df['h1'].isna()) & (m_df['h2'].isna()), 'h2'] = ''
    m_df.loc[(~m_df['h2'].isna()) & (m_df['h3'].isna()), 'h3'] = ''
    
    m_df[['h1', 'h2', 'h3']] = m_df[['h1', 'h2', 'h3']].fillna(method='ffill')
    
    return m_df


def group_by_doc(_df):
    df = _df.copy()
    
    # H4 단위로 Text 병합
    doc_merge_contents = df[df['ctype']=='text'].groupby('doc_group')['merge_contents'].apply(lambda x: '\n\n'.join(x)).reset_index()
    doc_merge_contents.columns = ['doc_group', 'doc_contents']
    
    doc_df = df[df['ctype']=='text'].merge(doc_merge_contents, on='doc_group', how='left').drop_duplicates('doc_group', keep='first')
    
    # DOC 내 이미지 ULR 추가
    doc_images = df[df['ctype']=='image']
    doc_images = doc_images.groupby('doc_group')['img_urls'].apply(lambda x: list(x)).reset_index()

    # DOC 내 테이블 URL 추가
    doc_tables = df[df['ctype']=='table']
    doc_tables = doc_tables[doc_tables['merge_contents'] != ''].copy()

    doc_tables = doc_tables.groupby('doc_group')[['merge_contents', 'img_urls', 'csv_urls']].apply(lambda x: pd.Series([x['merge_contents'].tolist(),x['img_urls'].tolist(), x['csv_urls'].tolist()])).reset_index()
    doc_tables.columns = ['doc_group', 'table_contents', 'table_img_urls', 'table_csv_urls']
    
    # DOC 병합
    final_doc_df = doc_df.drop(['img_urls', 'merge_contents'], axis=1).merge(doc_images, on='doc_group', how='left')\
                    .merge(doc_tables, on='doc_group', how='left')
                    
    final_doc_df = final_doc_df[~((final_doc_df['h1']==final_doc_df['doc_contents']) & (final_doc_df['table_contents'].isna()))]
    final_doc_df = final_doc_df[~((final_doc_df['h2']==final_doc_df['doc_contents']) & (final_doc_df['table_contents'].isna()))]
    final_doc_df = final_doc_df[~((final_doc_df['h3']==final_doc_df['doc_contents']) & (final_doc_df['table_contents'].isna()))]
    
    final_doc_df[['doc_contents']] = final_doc_df[['doc_contents']].fillna('')
    final_doc_df['table_contents'] = final_doc_df['table_contents'].apply(lambda d: d if np.array(d) is not np.nan else None)
    final_doc_df['img_urls'] = final_doc_df['img_urls'].apply(lambda d: d if np.array(d) is not np.nan else None)
    final_doc_df['table_img_urls'] = final_doc_df['table_img_urls'].apply(lambda d: d if np.array(d) is not np.nan else None)
    final_doc_df['table_csv_urls'] = final_doc_df['table_csv_urls'].apply(lambda d: d if np.array(d) is not np.nan else None)
    
    final_doc_df = final_doc_df.reset_index(drop=True)
    
    return final_doc_df

    
    