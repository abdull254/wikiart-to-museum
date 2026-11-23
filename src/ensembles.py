"""Cheap ensembles from saved predictions (no GPU, no training).

Averages class probabilities of
  - the 3 ResNet-50 seeds ("ResNet-50 seed ensemble")
  - ResNet-50 (3-seed mean) + CLIP probe + DINOv2 probe, equal weights (no tuning)
and scores them like the table of record: balanced accuracy, top-1, top-3, macro-F1,
painter-level bootstrap 95% CI, and a paired painter-level bootstrap against the best single member.
Museum rows = unseen artists. Also scores the Wikidata-labelled museum paintings separately.

Run from the repo root:  python src/ensembles.py
"""
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score as BA, f1_score

R = Path(__file__).resolve().parents[1]
C = ['Baroque', 'Mannerism (Late Renaissance)', 'Northern Renaissance', 'Realism', 'Rococo', 'Romanticism']
PC = [f'p_{c}' for c in C]
OUT = R / 'results/ensembles'; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

split = pd.read_csv(R / 'data/final/wikiart_split.csv')
mus = pd.read_csv(R / 'data/final/museum_test.csv')
mus['key'] = mus['museum'] + ':' + mus['id'].astype(str)

def base(set_name):
    """Reference table for a set: key, true style, painter, group, label source; in the probe files' order."""
    if set_name == 'test':
        d = pd.read_csv(R / 'results/probes/preds_clip_test.csv')[['image']]
        d = d.merge(split[['image', 'wikiart_idx', 'style', 'artist_key']], on='image', how='left')
        d['key'] = d['wikiart_idx'].astype(str); d['painter'] = 'w:' + d['artist_key'].astype(str); d['group'] = 'unseen'; d['source'] = 'WikiArt'
    else:
        d = pd.read_csv(R / f'results/probes/preds_clip_{set_name}.csv')[['image']]
        d = d.merge(mus[['image', 'key', 'style', 'artist', 'artist_group', 'source']], on='image', how='left')
        d = d.rename(columns={'artist_group': 'group'})
        anon = d['artist'].map(lambda a: (not isinstance(a, str)) or a.strip().lower() in ('', 'unknown', 'anonymous', 'anoniem', 'onbekend'))
        d['painter'] = np.where(anon, 'anon:' + d['key'], 'm:' + d['artist'].astype(str))   # same painter key as final_analysis.py
    assert d['key'].notna().all() and len(d) == d['key'].nunique(), set_name
    return d.reset_index(drop=True)

def probs(member, set_name, b):
    if member.startswith('resnet'):
        f = R / f'results/runs/resnet50_seed{member[-1]}/preds_{"wikiart_test" if set_name == "test" else set_name}.csv'
        d = pd.read_csv(f); d['key'] = d['id'].astype(str)
        d = b[['key']].merge(d, on='key', how='left')
    else:
        d = pd.read_csv(R / f'results/probes/preds_{member}_{set_name}.csv')
        assert (d['image'].values == b['image'].values).all()
    P = d[PC].values.astype(float); assert not np.isnan(P).any(), (member, set_name)
    return P / P.sum(1, keepdims=True)

def painter_index(painters):
    a = np.asarray(painters)
    return [np.where(a == u)[0] for u in np.unique(a)]

def boot(y, preds, groups, B=2000):
    out = np.empty((B, len(preds)))
    for b in range(B):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        out[b] = [BA(y[idx], p[idx]) for p in preds]
    return 100 * out

rows, paired = [], []
SETS = {'test': 'WikiArt test', 'met': 'Met', 'rijks': 'Rijksmuseum'}
for s, label in SETS.items():
    b = base(s); y = b['style'].map(C.index).values
    P = {m: probs(m, s, b) for m in ('resnet1', 'resnet2', 'resnet3', 'clip', 'dinov2')}
    P['ResNet-50 seed ensemble'] = (P['resnet1'] + P['resnet2'] + P['resnet3']) / 3
    P['ResNet-50 + CLIP + DINOv2'] = (P['ResNet-50 seed ensemble'] + P['clip'] + P['dinov2']) / 3
    P['CLIP + DINOv2'] = (P['clip'] + P['dinov2']) / 2
    names = {'ResNet-50 seed ensemble': 'ResNet-50 seed ensemble', 'clip': 'CLIP probe', 'dinov2': 'DINOv2 probe',
             'CLIP + DINOv2': 'CLIP + DINOv2', 'ResNet-50 + CLIP + DINOv2': 'ResNet-50 + CLIP + DINOv2'}
    subsets = {'unseen': b['group'].values == 'unseen'}
    if s != 'test':
        subsets['unseen, Wikidata label'] = subsets['unseen'] & (b['source'].values == 'artwork movement (Wikidata)')
    for grp, m in subsets.items():
        if m.sum() == 0: continue
        groups = painter_index(b['painter'].values[m]); yy = y[m]
        single_seed = np.mean([100 * BA(yy, P[f'resnet{k}'][m].argmax(1)) for k in (1, 2, 3)])
        preds = [P[k][m].argmax(1) for k in names]
        bs = boot(yy, preds, groups, B=1000 if grp == 'unseen' else 500)
        for j, (k, nm) in enumerate(names.items()):
            p = P[k][m]
            rows.append(dict(set=label, group=grp, model=nm, n=int(m.sum()), painters=len(groups), styles=len(np.unique(yy)),
                             bal_acc=100 * BA(yy, p.argmax(1)), ci_low=np.percentile(bs[:, j], 2.5), ci_high=np.percentile(bs[:, j], 97.5),
                             top1=100 * (yy == p.argmax(1)).mean(), top3=100 * (np.argsort(-p, 1)[:, :3] == yy[:, None]).any(1).mean(),
                             macro_f1=100 * f1_score(yy, p.argmax(1), average='macro'),
                             resnet_single_seed_mean=single_seed if nm == 'ResNet-50 seed ensemble' else np.nan))
        if grp == 'unseen':
            keys = list(names)
            for a, c in (('ResNet-50 seed ensemble', None), ('ResNet-50 + CLIP + DINOv2', 'clip'), ('CLIP + DINOv2', 'clip')):
                ia = keys.index(a)
                if c is None:   # seed ensemble vs single seeds: paired with each seed, report mean
                    continue
                ic = keys.index(c); d = bs[:, ia] - bs[:, ic]
                obs = rows[-len(keys) + ia]['bal_acc'] - rows[-len(keys) + ic]['bal_acc']
                paired.append(dict(set=label, model_a=names[a], model_b=names[c], diff=obs,
                                   ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                                   p=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
tab = pd.DataFrame(rows).round(2); pt = pd.DataFrame(paired).round(2)
tab.to_csv(OUT / 'ensembles.csv', index=False); pt.to_csv(OUT / 'ensembles_paired.csv', index=False)
pd.set_option('display.width', 250)
txt = ('ENSEMBLES FROM SAVED PREDICTIONS (equal-weight probability averages, nothing tuned)\n'
       'Balanced accuracy etc. in %; museum = painters not in WikiArt train; CI = painter-level bootstrap.\n\n'
       + tab.to_string(index=False) + '\n\nPAIRED (painter-level bootstrap, same resamples)\n' + pt.to_string(index=False) + '\n')
(OUT / 'ensembles_report.txt').write_text(txt); print(txt)
