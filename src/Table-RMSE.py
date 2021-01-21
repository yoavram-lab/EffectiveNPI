import os
import pandas as pd

no_fixed_free = ['7MNoTau', '7MFixed', '7M']
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
        df.loc[~idx, i] = ['{:.1f}'.format(x) for x in df.loc[~idx, i]] 
        df.loc[idx, i] = ['\\textbf{'+'{:.1f}'.format(x)+'}' for x in df.loc[idx, i]] 

df_out.set_index('Country', inplace=True)
bold_all(df_out, df_out.columns)
output_fname = '../figures/Table-RMSE.csv'
df_out.to_csv(output_fname,index='Country')
print(output_fname)