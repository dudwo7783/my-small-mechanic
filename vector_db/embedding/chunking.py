import pandas as pd
from itertools import chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ParentChildChunking():    
        
    def calculate_bucket_indices_pandas(self, input_series, bucket_size=1024):
        bucket_indices = []
        current_bucket_index = 0
        current_bucket_sum = 0

        for item in input_series:
            if current_bucket_sum + item > bucket_size:
                current_bucket_index += 1
                current_bucket_sum = item
            else:
                current_bucket_sum += item
            bucket_indices.append(current_bucket_index)

        return pd.Series(bucket_indices, name='chunk_group')
    
    def colelct_context_by_semantic(self, _df):
        df = _df.copy()
        df = df[['h1', 'h2', 'h3', 'index', 'doc_contents', 'img_urls', 'table_contents', 'table_img_urls', 'table_csv_urls']]
        df['contents_size'] = df['doc_contents'].apply(len)

        
        def collect_context(x):
            doc_contents = x['doc_contents'].dropna()
            doc_contents = '\n\n'.join(doc_contents)
            

            img_urls = x['img_urls'].dropna()
            if len(img_urls)==0:
                img_urls = []
            else:
                list(chain.from_iterable(img_urls.tolist()))
                img_urls = list(chain.from_iterable(img_urls.tolist()))
                img_urls = [img_url for img_url in img_urls]
                
            tbl_contents = x['table_contents'].dropna()
            if len(tbl_contents)==0:
                tbl_contents = []
            else:
                tbl_contents = list(chain.from_iterable(tbl_contents.tolist()))
                tbl_contents = [tbl_contents for tbl_contents in tbl_contents]
                
            tbl_img_urls = x['table_img_urls'].dropna()
            if len(tbl_img_urls)==0:
                tbl_img_urls = []
            else:
                tbl_img_urls = list(chain.from_iterable(tbl_img_urls.tolist()))
                tbl_img_urls = [tbl_img_url for tbl_img_url in tbl_img_urls]
                
            tbl_csv_urls = x['table_csv_urls'].dropna()
            if len(tbl_csv_urls)==0:
                tbl_csv_urls = []
            else:
                tbl_csv_urls = list(chain.from_iterable(tbl_csv_urls.tolist()))
                tbl_csv_urls = [tbl_img_url for tbl_img_url in tbl_csv_urls]
            
                
            return pd.Series({'doc_contents':doc_contents, 'img_urls':img_urls, 'table_contents': tbl_contents, 'table_img_urls':tbl_img_urls, 'table_csv_urls':tbl_csv_urls})

        df['h3_shift'] = df['h3'].shift(1)
        df['tmp_group'] = 0
        df.loc[df['h3'] != df['h3_shift'], 'tmp_group'] = 1
        df['tmp_group'] = df['tmp_group'].cumsum()
        
        df['chunk_group'] = df.groupby(['h1', 'h2', 'tmp_group'], sort=False, dropna=False)['contents_size']\
                              .apply(lambda x: self.calculate_bucket_indices_pandas(x)).values
        merge_contents = df.groupby(['h1','h2','h3', 'chunk_group']).apply(lambda x: collect_context(x)).reset_index()
        merge_contents.loc[merge_contents['h3']!='', 'h3'] = '[' + merge_contents.loc[merge_contents['h3']!='', 'h3'] + ']'
        merge_contents['doc_contents'] = merge_contents['h3'] + '\n' + merge_contents['doc_contents']
        merge_contents['contents_size'] = merge_contents['doc_contents'].apply(len)
        
        merge_contents['h2_shift'] = merge_contents['h2'].shift(1)
        merge_contents['tmp_group'] = 0
        merge_contents.loc[merge_contents['h2'] != merge_contents['h2_shift'], 'tmp_group'] = 1
        merge_contents['tmp_group'] = merge_contents['tmp_group'].cumsum()
        merge_contents['chunk_group2'] = merge_contents.groupby(['h1', 'h2', 'tmp_group'], sort=False, dropna=False)['contents_size']\
                                                       .apply(lambda x: self.calculate_bucket_indices_pandas(x)).values
        merge_contents = merge_contents.drop(columns=['h2_shift'])
        merge_contents_h3 = merge_contents.copy()
        merge_contents = merge_contents.groupby(['h1','h2', 'chunk_group2']).apply(lambda x: collect_context(x)).reset_index()
        merge_contents['contents_size'] = merge_contents['doc_contents'].apply(len)
        
        h3_tag = merge_contents_h3.groupby(['h1', 'h2', 'chunk_group2'])['h3'].apply(lambda x: ', '.join(x)).reset_index()
        h3_tag.loc[h3_tag['h3'].str.startswith(', '), 'h3'] = h3_tag.loc[h3_tag['h3'].str.startswith(', '), 'h3'].str.slice(start=2)
        merge_contents = merge_contents.merge(h3_tag, on=['h1', 'h2', 'chunk_group2'])
        
        return merge_contents
    
    def parent_child_chunking(self, _df, chunk_size=1024, overlap=50):
        df = _df.copy()
        child_text_splitter = RecursiveCharacterTextSplitter(
                            # Set a really small chunk size, just to show.
                            chunk_size=chunk_size,
                            chunk_overlap=overlap,
                            length_function=len,
                            is_separator_regex=False)
        
        
        df['child_chunk'] = df['doc_contents'].apply(
            lambda contents: list(map(lambda x: x.page_content, child_text_splitter.create_documents([contents]))))
        
        df = df.reset_index(names='doc_id')
        df = df.explode('child_chunk')
        
        return df

            