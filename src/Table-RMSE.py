import os
import pandas as pd

no_fixed_free = ['2020-06-27-Apr11-notau', '2020-06-25-Apr11-fixedtau', '2020-05-26-Apr11']
main_folder = r'../output'
nodf, fixeddf, freedf = [pd.read_csv(os.path.join(main_folder, inner_folder, 'figures', 'ppc_rmse.csv'),sep='\t',header=None) 
       for inner_folder in no_fixed_free]

df_out = pd.DataFrame()
df_out['Country'] = nodf[0]
df_out['No'] = nodf[2]
df_out['Fixed'] = fixeddf[2]
df_out['Free'] = freedf[2]

def bold_all(df, columns):
    minidxs = df.idxmin(axis=1)
    for i in columns:
        idx = i==minidxs
        df.loc[idx, i] = ['\\textbf{'+'{:.2f}'.format(x)+'}' for x in df.loc[idx, i]] 

df_out.set_index('Country', inplace=True)
bold_all(df_out, df_out.columns)
df_out.to_csv('../figures/Table-RMSE_tmp.csv',index='Country')