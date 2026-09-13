"""Stdlib-only provenance and document verification; never imports model code."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
LEDGER = Path('C:/Users/trhua/Documents/Codex/2026-09-08/f/PointMaze-ETT-experiment-ledger-f85a5f4.md')
HEAD = 'f85a5f44b176120333a1de0f83d6d0d02a2b290e'
SOURCES = [
    'crl/envs.py', 'scripts/collect_swamp_windy.py',
    'ett/action_reference.py', 'ett/diagonal_transition.py',
    'ett/pointmaze_region_pilot.py', 'ett/pointmaze_norm_comparison.py',
    'ett/report_pointmaze_norm_comparison.py', 'ett/run_convex_adversarial.py',
    'ett/rollout_return.py', 'propensity/nominal_policy.py',
    'artifacts/death_observability/f4_p30_expert_s01/REPORT.md',
    'artifacts/ett_death_admissibility/frozen_f4_gate_v1/REPORT.md',
    'artifacts/ett_death_admissibility/frozen_f4_gate_v1/DERIVATIONS.md',
    'artifacts/ett_structure_audit/fork_saved_s01_v1/REPORT.md',
    'artifacts/pointmaze_region_pilot/fixed_goal_h10_s01_v1/SUMMARY.md',
    'artifacts/pointmaze_region_pilot/update_reference_s01_v1/SUMMARY.md',
    'artifacts/pointmaze_region_pilot/native_policy_s01_v1/SUMMARY.md',
    'artifacts/pointmaze_region_pilot/full_f4_integration_s01_v1/REPORT.md',
    'artifacts/pointmaze_region_pilot/response_search_s01_v1/SUMMARY.md',
    'artifacts/pointmaze_region_pilot/response_search_s01_v1/candidate_pool.npz',
    'artifacts/pointmaze_region_pilot/response_search_s01_v1/manifest.json',
    'artifacts/pointmaze_region_pilot/continuation_precision_s01_v1/SUMMARY.md',
    'artifacts/pointmaze_region_pilot/contrast_agreement_v1/SUMMARY.md',
    'artifacts/pointmaze_region_pilot/pessimism_ceiling_v1/REPORT.md',
    'outputs/ett_frobenius_spectral_v1/PROTOCOL.md',
    'outputs/ett_frobenius_spectral_v1/results.json',
    'outputs/ett_frobenius_spectral_v1/completion.json',
    'outputs/ett_frobenius_spectral_v1/initial_parameters.npz',
    'outputs/ett_frobenius_spectral_v1/provenance.json',
    'artifacts/ett_rollout_return/residual6_s01/config.json',
]

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()

assert git('rev-parse', 'HEAD') == HEAD
assert git('branch', '--show-current') == 'feature/pointmaze-causal-transition'
assert git('diff', '--name-only') == ''
assert git('diff', '--cached', '--name-only') == ''
snapshot = OUT / 'ledger_before.md'
provenance_path = OUT / 'provenance.json'

if '--prepare' in sys.argv:
    assert not snapshot.exists() and not provenance_path.exists()
    snapshot.write_bytes(LEDGER.read_bytes())
    config = json.loads((ROOT / SOURCES[-1]).read_text(encoding='utf-8'))
    inherited = {}
    for label in ['dataset', 'nominal', 'actor', 'control']:
        relative = config['paths'][label]
        path = ROOT / relative
        inherited[label] = {'path': relative, 'exists': path.is_file()}
        if path.is_file():
            inherited[label]['sha256'] = digest(path)
            inherited[label]['matches_recorded'] = digest(path) == config['input_file_sha256'][relative]
            assert inherited[label]['matches_recorded'], relative
            SOURCES.append(relative)
    inputs = {relative: digest(ROOT / relative) for relative in SOURCES}
    candidate_manifest = json.loads((ROOT / 'artifacts/pointmaze_region_pilot/response_search_s01_v1/manifest.json').read_text())
    assert inputs['artifacts/pointmaze_region_pilot/response_search_s01_v1/candidate_pool.npz'] == candidate_manifest['candidate_pool.npz']
    write_json(provenance_path, {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'pointmaze_head': HEAD, 'remote_head_observed': HEAD,
        'remote_check': 'git ls-remote origin refs/heads/feature/pointmaze-causal-transition; successful read-only network escalation',
        'parent_checkout_head_observed': 'f290056c1559baedd98d4e5cc79cad3532b321b0',
        'ledger_original_path': str(LEDGER), 'ledger_before_sha256': digest(snapshot),
        'protocol_sha256': digest(OUT / 'PROTOCOL.md'),
        'timing': 'Protocol followed initial source/ledger reconnaissance; no new numerical experiment occurred.',
        'input_sha256': inputs, 'inherited_inputs': inherited,
        'scope': 'File hashes and source-level reasoning, not replay of historical rollouts or fitting.'})

provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
assert digest(snapshot) == provenance['ledger_before_sha256']
assert digest(OUT / 'PROTOCOL.md') == provenance['protocol_sha256']
for relative, expected in provenance['input_sha256'].items():
    assert digest(ROOT / relative) == expected, relative

before = snapshot.read_bytes()
addition = (OUT / 'LEDGER_ADDENDUM.md').read_bytes()
assert b'E27' not in before
combined = before + b'\r\n\r\n' + addition
candidate = OUT / 'ledger_updated.md'
if '--prepare' in sys.argv:
    candidate.write_bytes(combined)
assert candidate.read_bytes() == combined

if '--append-ledger' in sys.argv:
    current = LEDGER.read_bytes()
    assert current in (before, combined), 'Ledger changed independently: refuse overwrite.'
    if current == before:
        # User explicitly requested this append. Preserve the complete original byte prefix.
        with LEDGER.open('ab') as stream:
            stream.write(combined[len(before):])
    assert LEDGER.read_bytes() == combined

for name in ['PROTOCOL.md', 'REPORT.md', 'CONSTRUCTION.md', 'LEDGER_ADDENDUM.md']:
    content = (OUT / name).read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)', content):
        assert (OUT / target).exists(), (name, target)

write_json(OUT / 'verification.json', {
    'verified_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'pass', 'input_files_verified': len(provenance['input_sha256']),
    'tracked_worktree_and_index_unchanged': True,
    'source_candidate_pool_matches_manifest': True,
    'ledger_original_bytes_preserved': candidate.read_bytes().startswith(before),
    'external_ledger_updated': LEDGER.read_bytes() == combined,
    'local_markdown_links_exist': True,
    'new_native_transitions': 0, 'new_model_draws': 0, 'new_emitter_calls': 0,
    'new_ett_updates': 0, 'new_critic_updates': 0, 'new_actor_updates': 0,
    'new_nominal_updates': 0, 'numerical_experiment_executed': False,
    'gate': 'unvalidated_generated_branch_onset_construction',
    'H1': 'unresolved', 'H2': 'unresolved',
    'deliverable_sha256': {name: digest(OUT / name) for name in
        ['PROTOCOL.md', 'REPORT.md', 'CONSTRUCTION.md', 'LEDGER_ADDENDUM.md',
         'ledger_before.md', 'ledger_updated.md']}})
print(json.dumps({'status': 'pass', 'verified_inputs': len(provenance['input_sha256']),
                  'external_ledger_updated': LEDGER.read_bytes() == combined}))
