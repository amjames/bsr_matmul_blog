import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and merge data
all_data = pd.concat([
    pd.read_csv("torch-2.0_results.csv"),
    pd.read_csv("torch-2.1_results.csv")
], axis='index').dropna(axis=0, subset=['speedup']).drop(labels=['note'], axis=1)

# filter down batch sizes
all_data = all_data[all_data['n_batch'].isin([1, 8, 64, 256])]

# convert sparsity ratio to % materialized
all_data['Sparsity(%)'] = all_data['sparsity'] * 100
# Relabel column for plot
all_data['Speedup (larger is better)'] = all_data['speedup']

# figure generator for each dtype
def gen_dtype_figure(data, name):

    g = (sns.relplot(kind='line',
                     data=data,
                     x='Sparsity(%)',
                     y='Speedup (larger is better)',
                     hue='kernel',
                     row='size',
                     col='blocksize',
                     style='n_batch',
                     markers=True)
         .set_titles("sparse({row_name}, {row_name})[blocksize={col_name}] @ dense(n_batch, {row_name}, {row_name})")
        )
    for (row, col), ax in g.axes_dict.items():
        print(f"Decorating ax: {row}, {col}")
        ax.axhline(y=1.0, color='k', dashes=(2, 1))
        ax.axhline(y=10, color='k', dashes=(2, 1))
        ax.text(s='1x', x=55, y=1.1, color='k')
        ax.text(s='10x', x=55, y=10.1, color='k')
    g.figure.suptitle(f"dtype: {name}", size=15, y=1.1)
    return g.figure

all_data.groupby('dtype').apply(lambda g: gen_dtype_figure(g, g.name).savefig(f'{g.name}.png'))


