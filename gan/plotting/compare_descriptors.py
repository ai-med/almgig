# This file is part of Adversarial Learned Molecular Graph Inference and Generation (ALMGIG).
#
# ALMGIG is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ALMGIG is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ALMGIG. If not, see <https://www.gnu.org/licenses/>.
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import palettable
import pandas as pd
import seaborn as sns
from guacamol.utils.chemistry import continuous_kldiv, discrete_kldiv, get_fingerprints
from rdkit import Chem, DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm, trange

DESCRIPTORS = [
    'BertzCT',
    'MolLogP',
    'MolWt',
    'TPSA',
    'NumHAcceptors',
    'NumHDonors',
    'NumRotatableBonds',
    'NumAliphaticRings',
    'NumAromaticRings'
]


def read_smiles(filename, sep=None, skiprows=0):
    with open(filename) as fin:
        for i, line in enumerate(fin):
            if i < skiprows:
                continue
            smi, _ = line.split(sep, 1)
            yield smi


def cache(func):
    def inner(path, **kwargs):
        pkl = path.parent / (path.name + '_descriptors.pkl.gz')
        if pkl.exists():
            return pd.read_pickle(pkl)
        data = func(path, **kwargs)
        data.to_pickle(pkl)
        return data

    return inner


@cache
def load_training_data(train_file, name='Training'):
    train_data = pd.read_table(train_file, usecols=[0], squeeze=True, header=None)
    train_desc = get_descriptors(train_data, subset=train_data.shape[0] // 6)
    train_desc['type'] = name
    return train_desc


@cache
def load_predicted_data(predict_file):
    pred_data = pd.read_csv(predict_file, usecols=['SMILES'], squeeze=True)
    pred_desc = get_descriptors(pred_data)
    pred_desc['type'] = 'Generated'
    return pred_desc


def load_stats_data(json_file):
    with json_file.open() as fin:
        data = json.load(fin)
    return {v['benchmark_name']: v['score'] for v in data['results']}


def convert_text(name):
    s1 = re.sub('([a-z])([A-Z])', r'\1 \2', name)
    if s1.upper() != s1:
        s1 = re.sub('^([A-Z])([A-Z])', r'\1 \2', s1)
    return s1


def calculate_pc_descriptors(mols):
    output = []

    for i in mols:
        if i is not None:
            d = _calculate_pc_descriptors(i)
            output.append(d)

    return np.array(output)


def _calculate_pc_descriptors(mol):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTORS)

    _fp = calc.CalcDescriptors(mol)
    _fp = np.array(_fp)
    mask = np.isfinite(_fp)
    if (mask == 0).sum() > 0:
        print(f'{mol} contains an NAN physchem descriptor')
        _fp[~mask] = 0

    return _fp


def internal_similarity(mols):
    fps = get_fingerprints(mols)
    nfps = len(fps)

    similarities = np.empty((nfps, nfps), dtype=float)
    similarities[np.diag_indices(nfps)] = 0

    pbar = trange(1, nfps, desc='Computing internal similarity')
    for i in pbar:
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        similarities[i, :i] = sims
        similarities[:i, i] = sims

    return similarities.max(axis=1)


def get_descriptors(smiles, subset=None):  # assumes smiles list has no duplicates
    def to_mol(smi):
        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        return mol

    mols = [to_mol(x) for x in smiles]
    if subset is not None:
        mols = list(np.random.choice(mols, size=subset, replace=False))

    pbar = tqdm(mols, desc='Computing descriptors', total=len(mols))
    pred_desc = calculate_pc_descriptors(pbar)
    df = pd.DataFrame(pred_desc, columns=DESCRIPTORS)

    sim = internal_similarity(mols)
    df['InternalSimilarity'] = sim

    df_long = df.stack().reset_index(level=1)
    df_long.columns = ['descriptor', 'value']
    df_long.reset_index(drop=True, inplace=True)
    return df_long


def discrete_emd(hist_a, hist_b, bin_edges, norm=None):
    import ot
    from scipy.spatial.distance import cdist

    # normalize to 1 exactly to avoid warning:
    # Problem infeasible. Check that a and b are in the simplex
    hist_a = hist_a / hist_a.sum()
    hist_b = hist_b / hist_b.sum()
    # ground distance (substitution costs between bins)
    x = bin_edges[:-1, np.newaxis]
    M = cdist(x, x, 'euclidean')
    if norm is not None:
        M = norm(M)

    return ot.emd2(hist_a, hist_b, M)


def continuous_emd(samples_train, samples_gen, norm):
    # compute bins over all data
    _, bin_edges = np.histogram(np.concatenate((samples_train, samples_gen)),
                                bins='auto')
    hist_gen, _ = np.histogram(samples_gen, bins=bin_edges, density=True)
    hist_train, _ = np.histogram(samples_train, bins=bin_edges, density=True)

    return discrete_emd(hist_train, hist_gen, bin_edges, norm=norm)


class DescriptorsPlotter:

    def __init__(self, train_desc, pred_desc, dist, plot_color, palette=None):
        data = pd.concat((pred_desc, train_desc), axis=0, sort=True)
        data.reset_index(drop=True, inplace=True)
        self._grid = sns.FacetGrid(
            data, height=2.5, col='descriptor',
            col_order=['MolWt', 'MolLogP', 'BertzCT', 'TPSA', 'InternalSimilarity',
                       'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds', 'NumAliphaticRings',
                       'NumAromaticRings'],
            col_wrap=5, sharey=False, sharex=False,
            legend_out=False)

        self._dist = dist
        if dist == 'kl':
            self._dist_name = 'KL div'
        elif dist == 'emd':
            self._dist_name = 'EMD'
        else:
            raise ValueError('dist=%r is not supported' % dist)

        self._dist_scores = {}
        assert plot_color < len(palette), "palette has only {} elements, " \
                                          "but {} requested".format(len(palette), plot_color)
        self._pal = [palette[plot_color], palette[-1]]

    def _do_plot(self, data, color=None):
        nam = data['descriptor'].iloc[0]
        is_integer = nam.startswith('Num')

        groups = sorted(data['type'].unique())
        if is_integer:
            data['value'] = data['value'].astype(int)
        x = [data.query('type == @v').loc[:, 'value'] for v in groups]

        min_cost = min((np.std(v, ddof=1) for v in x)) + 1e-6

        # normalize ground distance by min of std dev
        def _norm_std(cost):
            return cost / min_cost

        if is_integer:
            bins = sorted(data['value'].unique())
            if bins[-1] > 10:
                data['value'] = np.clip(data['value'], 0, 10)
                bins = sorted(data['value'].unique())
                xticklabels = ['{:d}'.format(i) for i in bins[:-1]]
                xticklabels.append("10+")
            else:
                xticklabels = ['{:d}'.format(i) for i in bins]

            hists, bin_edges, _ = plt.hist(x, bins=bins, density=True, color=self._pal)
            plt.xticks(np.arange(len(bins)) + .5,
                       xticklabels)
            plt.gca().xaxis.grid(False)

            if self._dist == 'kl':
                dist_score = discrete_kldiv(x[-1], x[0])
            elif self._dist == 'emd':
                dist_score = discrete_emd(hists[-1], hists[0], bin_edges, _norm_std)

            # ax = sns.countplot(data=data, x='value', hue='type', palette='Set2')
        else:
            ax = sns.violinplot(data=data, x='descriptor', y='value', hue='type', split=True,
                                palette=self._pal)
            ax.set_xticklabels([''])

            if self._dist == 'kl':
                dist_score = continuous_kldiv(x[-1], x[0])
            elif self._dist == 'emd':
                dist_score = continuous_emd(x[-1], x[0], _norm_std)

        self._dist_scores[nam] = dist_score

    @property
    def dist_score(self):
        # Each value is transformed to be in [0, 1].
        # Then their average delivers the final score.
        partial_scores = [np.exp(-score) for score in self._dist_scores.values()]
        score = sum(partial_scores) / len(partial_scores)
        return score

    def plot(self):
        self._grid.map_dataframe(self._do_plot)
        self._grid.add_legend()
        for col, ax in zip(self._grid.col_names, self._grid.axes.flat):
            if col.startswith('Num'):
                name = convert_text(col[3:])
            else:
                name = convert_text(col)
            ax.set_title('{} ({} = {:.3f})'.format(name, self._dist_name, self._dist_scores[col]))


def plot_comparison(data: pd.DataFrame, hue, **kwargs):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax = sns.barplot(data=data, x='benchmark_name', y='score',
                     hue=hue, ax=ax, **kwargs)

    groups = kwargs.get('order')
    if groups is None:
        groups = data.loc[:, 'benchmark_name'].unique()
    hues = kwargs.get('hue_order')
    if hues is None:
        hues = data.loc[:, hue].unique()

    dd = data.set_index([hue, 'benchmark_name'])
    dd.sort_index(inplace=True)

    n_levels = len(hues)
    each_width = .8 / n_levels
    offsets = np.linspace(0, .8 - each_width, n_levels)
    offsets -= offsets.mean()

    for i, b in enumerate(groups):
        for j, a in enumerate(hues):
            value = dd.loc[(a, b), 'score']
            x = i + offsets[j]
            plt.text(x, 0.1, '{:.3f}'.format(value),
                     horizontalalignment='center',
                     rotation='vertical')

    ax.set_ylim(0., 1.)
    ax.set_ylabel('Score')
    ax.set_xlabel('')
    # put legend to the right of the plot
    ax.legend(bbox_to_anchor=(1.02, 0.75), loc='upper left',
              borderaxespad=0., title='Method')

    return ax


def plot_comparison_facet(data: pd.DataFrame, palette=None, order=None, hue_order=None,
                          xlabel=None, title_suffix=None):
    g = sns.FacetGrid(data=data,
                      col='benchmark_name', col_order=order,
                      sharex=True, sharey=True, aspect=data.loc[:, 'Method'].nunique() / 9.33333,
                      gridspec_kws={'wspace': 0.05},
                      legend_out=True)

    if hue_order is None:
        hues = data.loc[:, 'Method'].unique()
    else:
        hues = hue_order

    def _plot_bar(data, color=None, **kwargs):
        ax = sns.barplot(data=data, x='Method', y='score',
                         order=hue_order, palette=palette)
        dd = data.set_index('Method')
        for x, a in enumerate(hues):
            value = dd.loc[a, 'score']
            color = 'black' if value < 0.3 else 'white'
            plt.text(x, 0.125, '{:.3f}'.format(value),
                     horizontalalignment='center',
                     rotation='vertical',
                     color=color)

    g.map_dataframe(_plot_bar)
    title = '{col_name}'
    if title_suffix is not None:
        title += title_suffix
    g.set_titles(title, fontweight='bold')
    g.set_ylabels('Score')
    if xlabel is not None:
        g.set_xlabels(xlabel)

    from matplotlib.patches import Rectangle

    ax = g.axes[0, -1]
    p = [x for x in ax.get_children() if isinstance(x, Rectangle)]
    ax.legend(p, hues, bbox_to_anchor=(1.05, 1.0), loc='upper left',
              borderaxespad=0., title='Method')

    for ax in g.axes.flat:
        l, u = ax.get_ylim()
        ax.set_ylim(0, u)
        plt.setp(ax.get_xticklabels(), visible=False)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--train_file', type=Path, required=True)
    parser.add_argument('--predict_file', nargs='+', type=Path)
    parser.add_argument('--name', nargs='+', required=True)
    parser.add_argument('--distance', choices=['kl', 'emd'], required=True,
                        help="Whether to use KL divergence or Earth Mover's Distance.")
    parser.add_argument('-p', '--palette', choices=['ablation', 'stota'], required=True)

    args = parser.parse_args(args=args)

    if len(args.predict_file) != len(args.name):
        parser.error('--predict_file and --name must have the same length')

    sns.set_style('whitegrid')

    hue_order = args.name

    pal_sota = palettable.tableau.Tableau_10.mpl_colors
    del pal_sota[7]  # remove gray
    pal_abl = palettable.cartocolors.qualitative.Vivid_10.mpl_colors
    pal_abl[1] = pal_abl[0]
    pal_abl[0] = pal_sota[0]

    pal = pal_abl if args.palette == "ablation" else pal_sota
    pal.append(np.array([172, 97, 60]) / 255.)  # color for training data

    if 'Random' in hue_order:
        pal[hue_order.index('Random')] = (0.5, 0.5, 0.5)  # grey color for random

    if "_test" in args.train_file.name:
        ref_name = "Testing"
    else:
        ref_name = "Training"
    suffix = " wrt {}".format(ref_name)

    rename_pattern = re.compile(r"[^-_0-9A-Za-z]")

    train_desc = load_training_data(args.train_file, name=ref_name)
    stats_data = {'benchmark_name': [], 'score': [], 'Method': []}
    for i, (pred_file, name) in enumerate(zip(args.predict_file, args.name)):
        pred_desc = load_predicted_data(pred_file)
        pred_desc['type'] = name

        plotter = DescriptorsPlotter(train_desc, pred_desc,
                                     dist=args.distance,
                                     plot_color=i,
                                     palette=pal)
        plotter.plot()

        pdf_file = (args.output / rename_pattern.sub("_", name)).with_suffix('.pdf')
        plt.savefig(str(pdf_file), bbox_inches='tight')
        plt.close()

        stats = load_stats_data(pred_file.with_suffix('.json'))
        stats['Distribution Learning'] = plotter.dist_score
        for k, v in stats.items():
            stats_data['benchmark_name'].append(k)
            stats_data['score'].append(v)
            stats_data['Method'].append(name)

    stats_data = pd.DataFrame.from_dict(stats_data)

    order = ['Distribution Learning', 'Validity', 'Uniqueness', 'Novelty']

    plot_comparison_facet(data=stats_data.query('benchmark_name != "Distribution Learning"'),
                          order=order[1:], hue_order=hue_order, palette=pal)
    pdf_file = (args.output / ('comparison_simple_' + args.palette)).with_suffix('.pdf')
    plt.savefig(str(pdf_file), bbox_inches='tight')

    if args.distance == 'emd':
        xlabel = r'mean $-\exp(\mathrm{EMD}$)'
    elif args.distance == 'kl':
        xlabel = r'mean $-\exp(\mathrm{KLdiv}$)'

    plot_comparison_facet(data=stats_data.query('benchmark_name == "Distribution Learning"'),
                          hue_order=hue_order, palette=pal,
                          xlabel=xlabel, title_suffix=suffix)
    pdf_file = (args.output / ('comparison_dist_' + args.palette)).with_suffix('.pdf')
    plt.savefig(str(pdf_file), bbox_inches='tight')


if __name__ == '__main__':
    main()
