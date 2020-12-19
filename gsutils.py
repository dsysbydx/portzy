import pandas as pd

def read_gs(gs_url, row_th):
    df = pd.read_html(gs_url,)[0]
    df.columns = df.iloc[0]
    df = df[1:]
    df.dropna(axis=1, how='all',inplace=True)
    df.dropna(axis=0, thresh=row_th,inplace=True)
    return df

