import glob
import gzip
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
r = []
datasets = {}
for filename in glob.glob("./results/res*.pklz"):
    f = gzip.open(filename, "rb")
    res = pickle.load(f)
    f.close()
    r.append(res)
    datasets = list(res.keys())
    algos = list(res[datasets[0]].keys())
kPcts = list(sorted(r[0][datasets[0]][algos[0]][0].keys()))
if not os.path.exists("figures"):
    try:
        os.makedirs("figures")
    except:
        pass
print(algos)
# Plot precision at k for the simple trees
colors = {"Gini": '#1f77b4',
          "Entropy": '#d62728',
          "TreeRank": '#ffa512',
          "MetaAP": '#2ca02c'}
linestyles = {"Gini": (0, (1, 2)),
              "Entropy": (0, (3, 1, 1, 1)),
              "TreeRank":  (0, (1, 1)),
              "MetaAP": "solid"}
matplotlib.rcParams.update({'font.size': 10})
algos = ['Gini', 'Entropy', 'TreeRank', 'MetaAP']
pctsPos = [10, 30, 50]
rows = 1
columns = len(pctsPos)
fig = plt.figure(1, figsize=(columns * 5, rows * 2))
subplotNumber = 0
for pctPos in pctsPos:
    subplotNumber += 1
    ax = fig.add_subplot(rows, columns, subplotNumber)
    nbDatasets = 0
    miny, maxy = 100, 0
    for j, a in enumerate(algos):
        mr = []
        for kPct in kPcts:
            vals = []
            for da in datasets:
                if float(da.split("%")[0]) <= pctPos:
                    if j == 0 and kPct == kPcts[0]:
                        nbDatasets += 1
                    vals.append(np.mean(
                             [r[i][da][a][0][kPct][1] for i in range(len(r))]))
            mr.append(np.mean(vals))
        ax.plot(kPcts, mr, label=a, lw=4, color=colors[a],
                linestyle=linestyles[a])
        ax.grid(True)
        if min(mr) < miny:
            miny = min(mr)
        if max(mr) > maxy:
            maxy = max(mr)
    if pctPos == pctsPos[0]:
        ax.legend(ncol=4, loc="lower right", bbox_to_anchor=(2.2, 1.25))
    ax.set_ylim([miny-0.5, maxy+0.5])
    ax.set_xlim([1, 100])
    if pctPos == 50:
        ax.set_title(f'Mean results over all the {nbDatasets} datasets')
    if pctPos != 50:
        ax.set_title((f'Mean results over the {nbDatasets} datasets\n' +
                      f'with less than {pctPos}% of positives'))
    if pctPos == 10:
        ax.set_ylabel("Precision in the top rank")
fig.subplots_adjust(wspace=0.09, hspace=0.1)
plt.savefig('figures/pAtKtrees.pdf', bbox_inches="tight")
fig.clf()
plt.close(fig)
# Plot AP for the simple trees
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['hatch.linewidth'] = 4.0
hatch = ('/', '.', '\\', '')
pctsPos = [50, 40, 30, 20, 10]
labels = [str(p)+"%" for p in pctsPos]
index = np.arange(len(pctsPos))
fig = plt.figure(1, figsize=(13.5, 6))
bar_width = 0.95 / len(algos)
ax = fig.add_subplot(1, 1, 1)
for j, a in enumerate(algos):
    mr = []
    for pctPos in pctsPos:
        vals = []
        for da in datasets:
            if float(da.split("%")[0]) <= pctPos:
                vals.append(np.mean([r[i][da][a][2] for i in range(len(r))]))
        mr.append(np.mean(vals))
    ax.bar(index+j*bar_width, mr, bar_width, hatch=hatch[j],
           color=colors[a], alpha=1, label=a, zorder=0)
    for i, v in enumerate(mr):
        vText = "{:4.1f}".format(v)
        ax.text(i-0.114+j*bar_width, v+0.8, vText, color='black',
                size=15, zorder=1)
ax.set_xlim([index[0]-0.2, index[-1]+0.9])
ax.set_xticks(index+(len(algos)-1)*bar_width/2)
ax.set_ylim([15, 71])
ax.set_xticklabels(labels)
ax.set_ylabel("Average Precision")
ax.set_xlabel(("Percentage of positives examples"))
ax.legend(loc="upper right", ncol=2)
fig.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('figures/APtrees.pdf', bbox_inches="tight")
fig.clf()
plt.close(fig)

# Plot AP for the forests/boosting
colors = {"ForestEntropy": '#d62728',
          "XGBRanker": '#9467bd',
          "SGBAP": '#ff6ea0',
          "MetaAPForest": '#2ca02c'}
linestyles = {"ForestEntropy": (0, (3, 1, 1, 1)),
              "XGBRanker":  (0, (3, 2)),
              "SGBAP": (0, (1, 4)),
              "MetaAPForest": "solid"}
algos = ['SGBAP', 'ForestEntropy', "XGBRanker", 'MetaAPForest']
algosNames = ['SGBAP', 'EntropyForest', "XGBRanker", 'MetaAPForest']
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['hatch.linewidth'] = 4.0
hatch = ('/', '.', '\\', '')
pctsPos = [50, 40, 30, 20, 10]
labels = [str(p)+"%" for p in pctsPos]
index = np.arange(len(pctsPos))
fig = plt.figure(1, figsize=(13.5, 6))
bar_width = 0.95 / len(algos)
ax = fig.add_subplot(1, 1, 1)
for j, a in enumerate(algos):
    mr = []
    for pctPos in pctsPos:
        vals = []
        for da in datasets:
            if float(da.split("%")[0]) <= pctPos:
                vals.append(np.mean([r[i][da][a][2] for i in range(len(r))]))
        mr.append(np.mean(vals))
    ax.bar(index+j*bar_width, mr, bar_width, hatch=hatch[j],
           color=colors[a], alpha=1, label=algosNames[j], zorder=0)
    for i, v in enumerate(mr):
        vText = "{:4.1f}".format(v)
        ax.text(i-0.114+j*bar_width, v+0.8, vText, color='black',
                size=15, zorder=1)
ax.set_xlim([index[0]-0.2, index[-1]+0.9])
ax.set_xticks(index+(len(algos)-1)*bar_width/2)
ax.set_ylim([33, 93])
ax.set_xticklabels(labels)
ax.set_ylabel("Average Precision")
ax.set_xlabel(("Percentage of positives examples"))
ax.legend(loc="upper right", ncol=2)
fig.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('figures/APforestsboosting.pdf', bbox_inches="tight")
fig.clf()
plt.close(fig)

# Plot precision at k for the forests/boosting
colors = {"ForestEntropy": '#d62728',
          "XGBRanker": '#9467bd',
          "SGBAP": '#ff6ea0',
          "MetaAPForest": '#2ca02c'}
linestyles = {"ForestEntropy": (0, (3, 1, 1, 1)),
              "XGBRanker":  (0, (3, 2)),
              "SGBAP": (0, (1, 4)),
              "MetaAPForest": "solid"}
matplotlib.rcParams.update({'font.size': 10})
algos = ['SGBAP', 'ForestEntropy', "XGBRanker", 'MetaAPForest']
algosNames = ['SGBAP', 'EntropyForest', "XGBRanker", 'MetaAPForest']
pctsPos = [10, 30, 50]
rows = 1
columns = len(pctsPos)
fig = plt.figure(1, figsize=(columns * 5, rows * 2))
subplotNumber = 0
for pctPos in pctsPos:
    subplotNumber += 1
    ax = fig.add_subplot(rows, columns, subplotNumber)
    nbDatasets = 0
    miny, maxy = 100, 0
    for j, a in enumerate(algos):
        mr = []
        for kPct in kPcts:
            vals = []
            for da in datasets:
                if float(da.split("%")[0]) <= pctPos:
                    if j == 0 and kPct == kPcts[0]:
                        nbDatasets += 1
                    vals.append(np.mean(
                             [r[i][da][a][0][kPct][1] for i in range(len(r))]))
            mr.append(np.mean(vals))
        ax.plot(kPcts, mr, label=algosNames[j], lw=4, color=colors[a],
                linestyle=linestyles[a])
        ax.grid(True)
        if min(mr) < miny:
            miny = min(mr)
        if max(mr) > maxy:
            maxy = max(mr)
    if pctPos == pctsPos[0]:
        ax.legend(ncol=5, loc="lower right", bbox_to_anchor=(2.37, 1.00))
    ax.set_ylim([miny-0.5, maxy+0.5])
    ax.set_xlim([1, 100])
    if pctPos == 10:
        ax.set_ylabel("Precision in the top rank")
    ax.set_xlabel(("Percentage of positive examples used\n" +
                   "to compute the size of the top rank"))
fig.subplots_adjust(wspace=0.09, hspace=0.1)
plt.savefig('figures/pAtKforestsboosting.pdf', bbox_inches="tight")
fig.clf()
plt.close(fig)
