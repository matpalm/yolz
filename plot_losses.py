import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--losses-json', type=str)
parser.add_argument('--plot-png', type=str, default=None)
opts = parser.parse_args()
print("opts", opts)

raise Exception("switch to wandb")

import json
import seaborn as sns
import pandas as pd

with open(opts.losses_json, 'r') as f:
    losses = json.load(f)

df1 = pd.DataFrame()
df1['step'] = range(len(losses))
df1['variable'] = 'loss'
df1['value'] = losses

df2 = pd.DataFrame()
df2['step'] = range(len(losses))
df2['variable'] = 'ra loss'
df2['value'] = df1['value'].rolling(100).mean()

df = pd.concat([df1, df2], ignore_index=True)

plot_fname = opts.plot_png
if plot_fname is None:
    plot_fname = opts.losses_json.replace('json', 'png')

plot = sns.lineplot(data=df, x='step', y='value', hue='variable')
plot.figure.savefig(plot_fname)
