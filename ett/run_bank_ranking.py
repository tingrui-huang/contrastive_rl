"""Bounded frozen nearest-bank audit on recorded PointMaze F4 next states."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from ett.bank_ranking import (reconstruct_death, outcome_labels, nearest_scores,
    binary_metrics, describe, match_outcomes, pair_ranking)
from scripts.make_swamp_f4_failure_bank import file_sha, content_sha


SOURCE = Path('artifacts/ett_distribution_matching/f4_p30_s01_guarded')
OBJECTIVE = Path('artifacts/ett_objective_diagnostic/f4_p30_s01')
SCORER = Path('artifacts/failure_scoring/scope_b_f4_p30_s01')
PROTOCOL = dict(version=1, score='existing normalized full-F4 hard nearest-reference squared distance',
    dtype='float32, original JAX implementation', score_direction='smaller is more failure-like',
    tie_rule='first saved bank index; ranking ties receive 0.5; exact floating equality',
    population='original seed-0 source validation episodes minus ALL original 256 bank-source episodes',
    observation_rows='1..length-1: actual next states, including horizon endpoint; reset row 0 supplementary only',
    thresholds=None, seed=93171, bootstrap_replicates=1000,
    matching=dict(seed=93172, xy_tolerance=.25, previous_xy_tolerance=.5, time_tolerance=5,
        exact_variables=['source membership', 'commanded goal', 'newest XY cell'],
        distance='sum of XY, previous XY and time gaps divided by their respective tolerances',
        phases={'onset': 0, 'established_first': 3},
        reuse='no episode reused in either role within a phase; independent matching across phases',
        scope='approximate observational context matching, not conditional ETT or causal ground truth'),
    publication='aggregates, reference usage, anonymized score sequences and episode-set provenance only; no raw audit fields',
    training=False, reward_analysis=False, candidate_selection=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def frozen_inputs():
    config = read(SOURCE/'config.json')
    dataset, bank = Path(config['arguments']['dataset']), Path(config['arguments']['bank'])
    expected = config['dataset']['dataset']['sha256']
    if not dataset.exists() or not bank.exists():
        raise FileNotFoundError(f'Required current PointMaze files: {dataset} (SHA256 {expected}); {bank}')
    assert file_sha(dataset)==expected
    assert file_sha(bank)==config['references']['bank_file_sha256']
    assert content_sha(bank)==config['references']['bank_content_sha256']
    assert content_sha(dataset)==config['references']['dataset_content_sha256']
    manifest = read(str(dataset)+'.manifest.json')
    assert manifest['sha256']==expected
    with np.load(dataset, allow_pickle=False) as archive:
        obs, action = archive['obs'], archive['act']
        bits, died = archive['swamp_bits'], archive['entered_active_swamp']
        meta = json.loads(str(archive['meta']))
        lengths = archive['lengths'] if 'lengths' in archive else np.full(len(obs), obs.shape[1])
    assert meta['env_name']=='point_two_route_swamp_windy_f4_v0'
    assert meta['per_cell_swamp_prob']==.3 and meta['max_episode_steps']==50
    assert meta['merge']['main_index_range']==[0, 6000] and meta['merge']['bad_index_range']==[6000, 6600]
    assert len(obs)==6600 and obs.shape[2]==16
    assert np.isfinite(obs).all() and np.isfinite(action).all()
    assert np.all(action[:, -1]==0), 'terminal action row must be dummy'
    assert np.all(obs[:, :, 8:] == obs[0, 0, 8:]), 'fixed commanded goal required'
    with np.load(bank, allow_pickle=False) as archive:
        bank_ep, bank_row, vectors = archive['episode_id'], archive['failure_row'], archive['goals']
        bank_meta = json.loads(str(archive['meta']))
    assert np.array_equal(vectors, obs[bank_ep, bank_row, :8])
    assert bank_meta['source_content_sha256']==content_sha(dataset)
    validation = np.random.default_rng(0).permutation(len(obs))[:660]
    np.testing.assert_array_equal(validation, config['references']['full_source_validation_episode_ids'])
    kept = ~np.isin(bank_ep, validation)
    np.testing.assert_array_equal(bank_ep[kept], config['references']['kept_episode_ids'])
    np.testing.assert_array_equal(bank_row[kept], config['references']['kept_rows'])
    assert kept.sum()==233 and len(vectors)==256
    evaluation = np.sort(np.setdiff1d(validation, bank_ep))
    return config, dataset, bank, obs, action, bits, died, lengths, meta, vectors, bank_ep, bank_row, kept, validation, evaluation, bank_meta


def summarize(mask, ep, dead, distance, scores):
    result = binary_metrics(dead[mask], distance[mask])
    result.update(episodes=int(np.unique(ep[mask]).size), distance=describe(distance[mask]),
                  newest=describe(scores['newest_distance2'][mask]), history=describe(scores['history_distance2'][mask]))
    total = float(distance[mask].sum())
    result['history_fraction_of_total_distance'] = float(scores['history_distance2'][mask].sum()/total) if total>0 else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError('use a fresh output directory; historical artifacts are immutable')
    started = time.perf_counter()
    (config, dataset, bank, obs, action, bits, died, lengths, meta, vectors,
     bank_ep, bank_row, kept, validation, evaluation, bank_meta) = frozen_inputs()
    input_paths = [dataset, Path(str(dataset)+'.manifest.json'), bank, bank.with_name(bank.stem+'_manifest.json'),
        SOURCE/'config.json', SOURCE/'evaluation_contexts.json', SOURCE/'REPORT.md',
        OBJECTIVE/'config.json', OBJECTIVE/'REPORT.md', SCORER/'config.json', SCORER/'REPORT.md',
        Path('artifacts/death_observability/f4_p30_expert_s01/REPORT.md'),
        Path('crl/envs.py'), Path('scripts/collect_swamp_windy_f4.py'), Path('ett/failure_objectives.py')]
    hashes = {p.as_posix(): file_sha(p) for p in input_paths}
    # Preserve hashes of existing model files without deserializing or invoking models.
    for folder in [SOURCE, SCORER, Path('artifacts/ett_diagonal'), Path('artifacts/nominal_policy')]:
        hashes.update({p.as_posix(): file_sha(p) for p in folder.rglob('*.pkl')})
    prior_ep = np.unique(read(SOURCE/'evaluation_contexts.json')['episode_id'])
    provenance = dict(input_sha256=hashes, dataset=dataset.as_posix(), bank=bank.as_posix(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        environment=meta['env_name'], active_probability=.3, horizon=50,
        bank_original_episodes=bank_ep.tolist(), bank_original_rows=bank_row.tolist(),
        bank_retained_episodes=bank_ep[kept].tolist(), bank_retained_rows=bank_row[kept].tolist(),
        original_visible_pool_count=bank_meta['n_bank_full'], original_bank_count=len(vectors), retained_count=int(kept.sum()),
        evaluation_episodes=evaluation.tolist(), excluded_validation_bank_episodes=np.intersect1d(validation, bank_ep).tolist(),
        privileged_bank_selection=bank_meta['selection_uses_privileged_fields'],
        bank_composition=bank_meta['compose'], normalization=config['kernel'],
        prior_objective_episode_overlap=int(np.isin(evaluation, prior_ep).sum()),
        upstream_reuse='all episodes used by actor; original scorer test; expert subset previously used for observability/model evaluation; original 1024 objective contexts partly overlap',
        rejected_stale_artifact='worktree artifacts/swamp_windy_f4_failure_bank and root datasets have p=0.1 provenance; not used',
        source_sha256={p: file_sha(p) for p in ['ett/bank_ranking.py', 'ett/run_bank_ranking.py']})
    out.mkdir(parents=True)
    write(out/'protocol.json', PROTOCOL)
    write(out/'provenance.json', provenance)
    # Protocol and immutable input provenance are saved before any score is computed.
    death = reconstruct_death(obs, bits, died, lengths)
    ep = np.repeat(evaluation, lengths[evaluation]-1)
    row = np.concatenate([np.arange(1, lengths[e]) for e in evaluation])
    state, previous, goal = obs[ep, row, :8], obs[ep, row-1, :8], obs[ep, row, 8:]
    incoming_action = action[ep, row-1]
    dead, age = outcome_labels(ep, row, death)
    # Independent comparison with the established pre-state/fatal-transition audit.
    from ett.dataset import TransitionArrays, load_audit_labels
    old_arrays = TransitionArrays(previous, incoming_action, state, goal, ep, row-1)
    old_labels = load_audit_labels(dataset, old_arrays)
    np.testing.assert_array_equal(dead, old_labels['post_death'] | old_labels['entering_death'])
    assert not np.isin(ep, bank_ep).any()
    source = np.where(ep<1200, 0, np.where(ep<6000, 1, 2))
    motion = np.linalg.norm(state[:, :2]-previous[:, :2], axis=1)
    waiting = np.max(np.abs(incoming_action), axis=1)<=1e-7
    low = motion<=.05
    equal = np.all(state.reshape(-1, 4, 2)==state[:, None, :2], axis=(1, 2))
    masks = dict(all=np.ones(len(ep), bool), alive=~dead, onset=age==0, age1=age==1, age2=age==2,
        established=age>=3, alive_waiting=(~dead)&waiting, alive_low_motion=(~dead)&low,
        alive_collision_proxy=(~dead)&low&(np.linalg.norm(incoming_action, axis=1)>.05),
        alive_equal_after_reset=(~dead)&equal, alive_moving=(~dead)&~low)
    matching = {}
    for name, death_age in PROTOCOL['matching']['phases'].items():
        matching[name] = match_outcomes(age==death_age, ~dead, ep, row, state, previous, source, goal, PROTOCOL['matching'])
    # Matching above cannot inspect scores. Full trajectory signatures are visible-only.
    signature = lambda e: hashlib.sha256(obs[e, :lengths[e]].tobytes()+action[e, :lengths[e]-1].tobytes()).hexdigest()
    bank_signatures = {signature(e) for e in bank_ep}
    clone = np.array([signature(e) in bank_signatures for e in evaluation])
    counts = dict(evaluation_episodes=len(evaluation), outcome_rows=len(ep), original_validation_episodes=len(validation),
        excluded_original_bank_episodes=len(np.intersect1d(validation, bank_ep)),
        reset_rows_supplementary=len(evaluation), terminal_next_rows=int(np.sum(row==lengths[ep]-1)),
        death_at_final_transition_episodes=int(np.sum(death[evaluation]==lengths[evaluation]-1)),
        death_episodes=int(np.sum(death[evaluation]>=1)),
        censored_before_age3_episodes=int(np.sum((death[evaluation]>=1)&(death[evaluation]+3>=lengths[evaluation]))),
        bank_trajectory_clone_episodes=int(clone.sum()),
        repeated_evaluation_trajectory_episodes=len(evaluation)-len({signature(e) for e in evaluation}),
        temporal_checks='exact F4 shift, seeded reset, absorbing XY, age3 equal stack, independent legacy label agreement',
        next_row_change_from_prior='score obs[t+1] at rows 1..50; prior classifier scored obs[t] at rows 0..49',
        no_recorded_collision_flags=True, rewards_not_recorded=True)
    write(out/'population.json', counts)
    write(out/'matching_coverage_before_scoring.json', {k: v[1] for k, v in matching.items()})
    print(f'Protocol fixed; {len(evaluation)} episodes / {len(ep)} recorded next outcomes. Scoring frozen bank.', flush=True)
    scores = nearest_scores(state, vectors[kept], config['kernel']['mean'], config['kernel']['std'])
    distance = scores['distance2']
    np.testing.assert_allclose(distance, scores['newest_distance2']+scores['history_distance2'], rtol=0, atol=0)
    assert all(np.isfinite(v).all() for v in scores.values())
    bank_bytes = {v.tobytes() for v in vectors}
    exact = np.array([s.tobytes() in bank_bytes for s in state])
    _, unique_rows = np.unique(state, axis=0, return_index=True)
    unique_mask = np.zeros(len(ep), bool); unique_mask[unique_rows] = True
    masks.update(excluding_exact_bank_vectors=~exact, unique_visible_vectors=unique_mask,
        outside_prior_objective_episodes=~np.isin(ep, prior_ep),
        excluding_bank_trajectory_clones=~np.isin(ep, evaluation[clone]))
    for s in range(3):
        masks[f'source_{s}'] = source==s
    cells = np.floor(state[:, :2]).astype(int)
    for x in [3, 4, 5]:
        masks[f'swamp_cell_{x}_3'] = np.all(cells==[x, 3], axis=1)
    masks['outside_swamp_cells'] = ~np.any(np.stack([masks[f'swamp_cell_{x}_3'] for x in [3, 4, 5]]), axis=0)
    for start in range(1, 51, 10):
        masks[f'time_{start}_{start+9}'] = (row>=start)&(row<start+10)
    results = {k: summarize(mask, ep, dead, distance, scores) for k, mask in masks.items()}
    comparisons = {}
    for positive in ['onset', 'age1', 'age2', 'established']:
        for negative in ['alive', 'alive_waiting', 'alive_low_motion', 'alive_collision_proxy', 'alive_moving']:
            selected = masks[positive]|masks[negative]
            comparisons[f'{positive}_vs_{negative}'] = binary_metrics(dead[selected], distance[selected])
        for subgroup in ['source_0','source_1','source_2','swamp_cell_3_3','swamp_cell_4_3','swamp_cell_5_3',
                         'time_1_10','time_11_20','time_21_30','time_31_40','time_41_50',
                         'excluding_exact_bank_vectors','unique_visible_vectors','outside_prior_objective_episodes']:
            selected = (masks[positive]|~dead)&masks[subgroup]
            comparisons[f'{positive}_vs_alive__{subgroup}'] = binary_metrics(dead[selected], distance[selected])
    decomposition = {}
    for positive in ['onset', 'established']:
        selected = masks[positive]|~dead
        decomposition[positive] = {k: binary_metrics(dead[selected], scores[k][selected])
                                  for k in ['distance2','newest_distance2','history_distance2']}
    matched = {}
    for name, (pairs, coverage) in matching.items():
        result = dict(coverage=coverage, ranking=pair_ranking(pairs, distance),
                      component_rankings={k: pair_ranking(pairs, scores[k]) for k in ['newest_distance2','history_distance2']})
        result['by_alive_stratum'] = {k: pair_ranking(pairs[masks[k][pairs[:, 1]]], distance)
            for k in ['alive_waiting','alive_low_motion','alive_collision_proxy','alive_moving']}
        result['by_source'] = {str(s): pair_ranking(pairs[source[pairs[:, 0]]==s], distance) for s in range(3)}
        result['excluding_exact_bank_vectors'] = pair_ranking(pairs[~exact[pairs].any(1)], distance)
        matched[name] = result
    reset_scores = nearest_scores(obs[evaluation, 0, :8], vectors[kept], config['kernel']['mean'], config['kernel']['std'])
    reset = dict(rows=len(evaluation), episodes=len(evaluation), alive_by_reset_semantics=True,
                 actual_next_state=False, distance=describe(reset_scores['distance2']))
    # Identical visible vectors can have opposite truth; report rather than erase them.
    _, inverse = np.unique(state, axis=0, return_inverse=True)
    dcount = np.bincount(inverse, weights=dead); acount = np.bincount(inverse, weights=~dead)
    overlap = dict(exact_original_bank_vector_rows=int(exact.sum()), episodes=int(np.unique(ep[exact]).size),
        dead_rows=int((exact&dead).sum()), alive_rows=int((exact&~dead).sum()),
        duplicate_visible_rows=len(state)-len(unique_rows),
        distinct_vectors_with_conflicting_labels=int(np.sum((dcount>0)&(acount>0))),
        exact_score_tie_rows=int(np.sum(scores['exact_tie_count']>1)),
        sensitivity='original bank (all 256 vectors) exclusion and first chronological unique visible vector; latter changes population weights')
    usage = []
    for index in range(int(kept.sum())):
        here = scores['index']==index
        usage.append(dict(index=index, source_episode=int(bank_ep[kept][index]), source_row=int(bank_row[kept][index]),
            selected_rows=int(here.sum()), selected_episodes=int(np.unique(ep[here]).size),
            alive_rows=int((here&~dead).sum()), onset_rows=int((here&(age==0)).sum()),
            established_rows=int((here&(age>=3)).sum())))
    # Real within-episode onset sequences, without publishing raw audit fields.
    onset_sequences = []
    for e in evaluation[death[evaluation]>=1]:
        lookup = {int(r): i for i, r in zip(np.flatnonzero(ep==e), row[ep==e])}
        values = []
        for a in range(5):
            i = lookup.get(int(death[e]+a))
            values.append(float(distance[i]) if i is not None else None)
        onset_sequences.append(values)
    complete = np.asarray([v for v in onset_sequences if all(x is not None for x in v)])
    sequence_summary = dict(ages=list(range(5)), available_counts=[sum(v[a] is not None for v in onset_sequences) for a in range(5)],
        available_distance=[describe([v[a] for v in onset_sequences if v[a] is not None]) for a in range(5)],
        complete_episodes=len(complete), complete_distance=[describe(complete[:, a]) for a in range(5)],
        paired_change_from_age0=[describe(complete[:, a]-complete[:, 0]) for a in range(5)],
        examples=complete[:6].tolist(), example_selection='first six complete onset episodes in ascending source order, score-blind; IDs and raw audit fields omitted')
    write(out/'metrics.json', dict(strata=results, comparisons=comparisons, component_diagnostics=decomposition,
        matched=matched, reset=reset, overlap=overlap, sequences=sequence_summary))
    write(out/'reference_usage.json', usage)
    # Export only aggregate plot data, never per-row privileged labels or masks.
    edges = np.linspace(-8, 3, 100)
    plot_data = {k: np.histogram(np.log10(np.maximum(distance[masks[k]], 1e-8)), edges)[0].tolist()
                 for k in ['onset','age1','age2','established','alive_waiting','alive_collision_proxy','alive_moving']}
    write(out/'plot_data.json', dict(log10_distance_edges=edges.tolist(), histograms=plot_data,
        underflow_overflow={k: dict(below=int(np.sum(distance[masks[k]]<1e-8)), above=int(np.sum(distance[masks[k]]>1e3))) for k in plot_data}))
    assert all(file_sha(p)==digest for p, digest in hashes.items())
    write(out/'verification.json', dict(status='passed', input_files_and_existing_models_unchanged=True,
        independent_label_alignment=True, bank_source_episodes_excluded=True,
        same_selected_reference_decomposition_exact=True, score_finite=True,
        matching_score_blind=True, matching_episode_reuse_absent=True,
        published_raw_privileged_fields=False, no_model_updates=True,
        elapsed_seconds=time.perf_counter()-started))
    print(json.dumps({k: v['ranking'] for k, v in matched.items()}, indent=2), flush=True)


if __name__=='__main__':
    main()
