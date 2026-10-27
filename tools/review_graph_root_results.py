"""Read-only follow-up: fixed thresholds, root controls, all-system graph effects.

Post-source exploratory analysis, not a preregistered significance gate. Inputs
are collected JSON results, not benchmark arrays. No torch/GPU dependency.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

METRICS = ['AP', 'FPR', 'conditional_Top1', 'joint_detected_root_recall']


def means(rows, metric):
    values = defaultdict(list)
    for r in rows:
        if r.get(metric) is not None:
            values[r['key']].append(r[metric])
    return {k: float(np.mean(v)) for k, v in values.items()}


def summarize(rows):
    return {m: float(np.mean(list(means(rows, m).values())))
            for m in METRICS if means(rows, m)}


def test_deltas(d):
    keys = sorted(d)
    v = np.array([d[k] for k in keys])
    rng = np.random.default_rng(348)
    boot = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
    perm = (rng.choice([-1, 1], (20000, len(v))) * v).mean(1)
    return dict(mean=float(v.mean()), CI95=np.quantile(boot, [.025, .975]).tolist(),
                p_signflip=float((1+(np.abs(perm)>=abs(v.mean())).sum())/20001),
                up=int((v>0).sum()), down=int((v<0).sum()), systems=len(v),
                all_system_differences={k: d[k] for k in keys})


def paired(a, b, metric):
    aa, bb = means(a, metric), means(b, metric)
    assert aa.keys() == bb.keys()
    return test_deltas({k: aa[k]-bb[k] for k in aa})


def main(a):
    assert not a.output.exists(), 'Preserve earlier analysis'
    prior = json.loads(a.prior.read_text())['all_rows']
    rows, fixed = defaultdict(list), defaultdict(list)
    files_verified = 0
    exposure = 0
    invariance = 0
    for port, graph in [('2226', 'true'), ('2229', 'wrong')]:
        folder = a.diagnostics/port
        col = json.loads((folder/'collection.json').read_text())
        for item in col['files']:
            p = folder/item['relative']
            assert hashlib.sha256(p.read_bytes()).hexdigest() == item['SHA256']
            files_verified += 1
        for path in (folder/'records').glob('*/result.json'):
            d = json.loads(path.read_text())
            assert d['optimizer_steps'] == d['new_basis_forwards'] == d['new_TimeRCD_forwards'] == 0
            assert not d['target_arrays_opened']
            exposure += d['cached_head_input_scalars']
            task = d['task']
            if task['kind'] == 'graph_cross':
                train, arm, seed = Path(task['checkpoint']['path']).name.split('__')
                seed = int(seed.split('_')[0][1:])
                name = f'{train}_train_{graph}_infer::{arm}'
                rows[name].extend(dict(r, optimization_seed=seed) for r in d['result']['metrics'])
                fixed[name].extend(dict(r, optimization_seed=seed) for r in d['original_threshold_metrics'])
            elif task['kind'] == 'root_readout':
                assert d['detection_scores_reused_exact']
                grouped = defaultdict(dict)
                for r in d['result']['metrics']:
                    rows[graph+'::root::'+r['method']].append(r)
                    grouped[(r['key'], r['view'])][r['method']] = r
                for group in grouped.values():
                    for method in ['fault_only', 'fault_vs_normal']:
                        for key in ['AP','PR_AUC','ROC_AUC','FPR','case_mean_score','case_max_score','event_recall']:
                            assert group['original'].get(key) == group[method].get(key)
                    invariance += 1
    assert files_verified == 42 and exposure == 242221056
    tests = {}
    for metric in ['conditional_Top1','joint_detected_root_recall']:
        tests['fault_only_minus_temporal::'+metric] = paired(
            rows['true::root::fault_only'], prior['true::stat_temporal::final'], metric)
    for metric in ['AP', 'conditional_Top1']:
        true = paired(prior['true::stat_temporal::final'], rows['true_train_wrong_infer::stat_temporal'], metric)
        wrong = paired(rows['wrong_train_true_infer::stat_temporal'], prior['wrong::stat_temporal::final'], metric)
        tests['true_training_additional_graph_effect::'+metric] = test_deltas({
            k: v-wrong['all_system_differences'][k] for k,v in true['all_system_differences'].items()})
    groups = {}
    for column in ['fault','background','C','family']:
        values = sorted({str(r[column]) for r in prior['true::stat_temporal::final']})
        for value in values:
            first = [r for r in prior['true::stat_temporal::final'] if str(r[column])==value]
            second = [r for r in rows['true_train_wrong_infer::stat_temporal'] if str(r[column])==value]
            if not means(first, 'AP'):continue
            groups[column+'='+value] = dict(correct=summarize(first), wrong=summarize(second),
                                            AP_difference=paired(first,second,'AP'))
    fixed_tests = {}
    for arm in ['stat_temporal','stat_pair','protected','raw_adapt']:
        name='true_train_wrong_infer::'+arm
        for metric in ['FPR','joint_detected_root_recall']:
            if means(prior['true::'+arm+'::final'],metric):
                fixed_tests[arm+'::'+metric]=paired(prior['true::'+arm+'::final'], fixed[name],metric)
    result = dict(status='complete', files_SHA_verified=files_verified,
                  root_metric_invariance_checks=invariance,
                  actual_cached_head_scalars=exposure, additional_training_exposure=0,
                  additional_GPU_compute=0, target_arrays_read=False,
                  fixed_original_threshold_summary={k:summarize(v) for k,v in fixed.items()},
                  separately_calibrated_summary={k:summarize(v) for k,v in rows.items()},
                  fixed_threshold_paired=fixed_tests, paired_tests=tests, temporal_graph_strata=groups,
                  limitations=['Exploratory; no multiplicity-adjusted confirmation',
                               '32 systems, seeds averaged within system',
                               'Conditional Top1 is not automatic localization success'],
                  prior_SHA256=hashlib.sha256(a.prior.read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status='complete',files=files_verified,root_checks=invariance,
                          tests={k:{x:v[x] for x in ['mean','CI95','p_signflip']} for k,v in tests.items()},
                          fixed_threshold_temporal=result['fixed_original_threshold_summary']['true_train_wrong_infer::stat_temporal'])))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--diagnostics',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
