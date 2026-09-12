"""Replay the frozen bank audit without publishing per-row privileged truth."""
import argparse
from pathlib import Path
import numpy as np

from ett.bank_ranking import (reconstruct_death, outcome_labels, nearest_scores,
    binary_metrics, match_outcomes, pair_ranking)
from ett.run_bank_ranking import frozen_inputs, PROTOCOL, read, write, summarize
from scripts.make_swamp_f4_failure_bank import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args(); root = Path(args.run_dir)
    saved = read(root/'metrics.json'); provenance = read(root/'provenance.json')
    assert read(root/'protocol.json')==PROTOCOL
    for p, digest in {**provenance['input_sha256'], **provenance['source_sha256']}.items():
        assert file_sha(p)==digest, p
    (config, _, _, obs, action, bits, died, lengths, _, vectors, bank_ep,
     _, kept, _, episodes, _) = frozen_inputs()
    death = reconstruct_death(obs, bits, died, lengths)
    ep = np.repeat(episodes, lengths[episodes]-1)
    row = np.concatenate([np.arange(1, lengths[e]) for e in episodes])
    state, previous, goal = obs[ep, row, :8], obs[ep, row-1, :8], obs[ep, row, 8:]
    dead, age = outcome_labels(ep, row, death)
    assert not np.isin(ep, bank_ep).any()
    score = nearest_scores(state, vectors[kept], config['kernel']['mean'], config['kernel']['std'])
    distance = score['distance2']
    assert summarize(np.ones(len(ep), bool), ep, dead, distance, score)==saved['strata']['all']
    for name, mask in [('onset', age==0), ('age1', age==1), ('age2', age==2), ('established', age>=3)]:
        assert summarize(mask, ep, dead, distance, score)==saved['strata'][name]
        selected = mask|~dead
        assert binary_metrics(dead[selected], distance[selected])==saved['comparisons'][name+'_vs_alive']
    source = np.where(ep<1200, 0, np.where(ep<6000, 1, 2))
    for name, death_age in PROTOCOL['matching']['phases'].items():
        pairs, info = match_outcomes(age==death_age, ~dead, ep, row, state, previous, source, goal, PROTOCOL['matching'])
        assert info==saved['matched'][name]['coverage']
        assert pair_ranking(pairs, distance)==saved['matched'][name]['ranking']
        f, a = pairs.T
        assert len(np.unique(ep[pairs]))==pairs.size
        assert np.all(source[f]==source[a]) and np.all(goal[f]==goal[a])
        assert np.all(np.floor(state[f, :2])==np.floor(state[a, :2]))
        assert np.all(np.abs(row[f]-row[a])<=5)
        assert np.all(np.linalg.norm(state[f, :2]-state[a, :2], axis=1)<=.25)
        assert np.all(np.linalg.norm(previous[f, :2]-previous[a, :2], axis=1)<=.5)
    # Independent NumPy arithmetic on deterministic observed rows, no score selection.
    selected = np.arange(0, len(state), max(1, len(state)//128))
    diff = (state[selected, None]-vectors[kept])/np.asarray(config['kernel']['std'], np.float32)
    newest = np.square(diff[:, :, :2]).sum(-1); history = np.square(diff[:, :, 2:]).sum(-1)
    values = newest+history; index = values.argmin(1)
    np.testing.assert_array_equal(index, score['index'][selected])
    np.testing.assert_allclose(values[np.arange(len(selected)), index], distance[selected], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(distance, score['newest_distance2']+score['history_distance2'])
    write(root/'replay_checks.json', dict(status='passed', primary_aggregate_metrics_exact=True,
        matching_and_ranking_exact=True, independent_numpy_score_check=True,
        matching_tolerances_and_episode_exclusion=True, immutable_input_hashes_match=True,
        raw_privileged_fields_published=False, unit_test_command='python -m unittest scripts.test_bank_ranking',
        unit_tests_passed=7, source_sha256=file_sha(__file__)))
    print('Passed aggregate and matching replay, independent score arithmetic and immutable provenance.')


if __name__=='__main__':
    main()
