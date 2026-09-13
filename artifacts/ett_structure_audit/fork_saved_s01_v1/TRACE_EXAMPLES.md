# Deterministically selected fork examples

Selection: first reset seed, then zero-based pre-action step t, within each named category. These categories describe the saved decomposition; they do not label hidden outcomes or causal correctness. Complete precision, gates and matrices are retained in selected_examples.json and fork_traces.json.

The clipping-blocks-entry category is empty in both seeds. The radius-obstacle category is empty in seed 0. These are retained as null entries, not replaced with selected near misses.

## Seed 0 / first

Reset 51000001, t=1.

- q: 1.494892, 3.545814
- execution_action: 0.005108, -1.000000
- nominal_action: 0.898952, -0.201605
- anchor: 2.405497, 3.352175
- anchor_displacement: 0.910605, -0.193638
- correction: -0.060476, 0.041347
- proposal: 2.345021, 3.393522
- rectangle: 0
- box_low: 0.494892, 3.000000
- box_high: 2.494892, 4.000000
- emitted: 2.345021, 3.393522
- emitted_displacement: 0.850129, -0.152292
- current_class_min_y: 3.000000
- all_maps_min_y: 2.545814

## Seed 0 / upward_before_clip

Reset 51000002, t=1.

- q: 1.492904, 3.324386
- execution_action: 0.007096, -1.000000
- nominal_action: 0.839311, 0.495395
- anchor: 2.338480, 3.826847
- anchor_displacement: 0.845576, 0.502461
- correction: -0.016098, 0.112533
- proposal: 2.322383, 3.939380
- rectangle: 0
- box_low: 0.492904, 3.000000
- box_high: 2.492904, 4.000000
- emitted: 2.322383, 3.939380
- emitted_displacement: 0.829478, 0.614994
- current_class_min_y: 3.000000
- all_maps_min_y: 2.324386

## Seed 0 / overlap_selector_exclusion

Reset 51000004, t=1.

- q: 1.406821, 3.514562
- execution_action: 0.093179, -1.000000
- nominal_action: 0.001086, -0.003895
- anchor: 1.402350, 3.502584
- anchor_displacement: -0.004472, -0.011978
- correction: 0.060712, 0.081012
- proposal: 1.463062, 3.583596
- rectangle: 0
- box_low: 0.406821, 3.000000
- box_high: 2.406821, 4.000000
- emitted: 1.463062, 3.583596
- emitted_displacement: 0.056240, 0.069034
- current_class_min_y: 3.000000
- all_maps_min_y: 2.514562

## Seed 0 / feasible_current_class_entry

Reset 51000049, t=2.

- q: 1.640548, 3.427011
- execution_action: -0.140548, -1.000000
- nominal_action: -0.072660, -0.633336
- anchor: 1.542955, 2.818495
- anchor_displacement: -0.097593, -0.608516
- correction: 0.010902, 0.023847
- proposal: 1.553858, 2.842342
- rectangle: 2
- box_low: 1.000000, 2.427011
- box_high: 2.000000, 4.000000
- emitted: 1.553858, 2.842342
- emitted_displacement: -0.086691, -0.584670
- current_class_min_y: 2.445600
- all_maps_min_y: 2.445600

## Seed 1 / first

Reset 51000001, t=1.

- q: 1.492248, 3.546925
- execution_action: 0.007752, -1.000000
- nominal_action: 0.898961, -0.202672
- anchor: 2.402870, 3.352216
- anchor_displacement: 0.910622, -0.194709
- correction: 0.049761, 0.065096
- proposal: 2.452631, 3.417311
- rectangle: 0
- box_low: 0.492248, 3.000000
- box_high: 2.492248, 4.000000
- emitted: 2.452631, 3.417311
- emitted_displacement: 0.960383, -0.129613
- current_class_min_y: 3.000000
- all_maps_min_y: 2.546925

## Seed 1 / upward_before_clip

Reset 51000002, t=1.

- q: 1.500000, 3.317726
- execution_action: 0.000000, -1.000000
- nominal_action: 0.838449, 0.501736
- anchor: 2.344727, 3.826453
- anchor_displacement: 0.844727, 0.508727
- correction: 0.047726, 0.079214
- proposal: 2.392453, 3.905667
- rectangle: 0
- box_low: 0.500000, 3.000000
- box_high: 2.500000, 4.000000
- emitted: 2.392453, 3.905667
- emitted_displacement: 0.892453, 0.587941
- current_class_min_y: 3.000000
- all_maps_min_y: 2.317726

## Seed 1 / overlap_selector_exclusion

Reset 51000004, t=1.

- q: 1.388743, 3.514241
- execution_action: 0.111257, -1.000000
- nominal_action: 0.001103, -0.003888
- anchor: 1.384321, 3.502314
- anchor_displacement: -0.004422, -0.011927
- correction: -0.010131, 0.001739
- proposal: 1.374190, 3.504053
- rectangle: 0
- box_low: 0.388743, 3.000000
- box_high: 2.388742, 4.000000
- emitted: 1.374190, 3.504053
- emitted_displacement: -0.014552, -0.010188
- current_class_min_y: 3.000000
- all_maps_min_y: 2.514241

## Seed 1 / feasible_current_class_entry

Reset 51000049, t=2.

- q: 1.621721, 3.437507
- execution_action: -0.121721, -1.000000
- nominal_action: -0.059806, -0.632324
- anchor: 1.537257, 2.830066
- anchor_displacement: -0.084464, -0.607441
- correction: 0.003007, 0.007375
- proposal: 1.540264, 2.837442
- rectangle: 2
- box_low: 1.000000, 2.437507
- box_high: 2.000000, 4.000000
- emitted: 1.540264, 2.837442
- emitted_displacement: -0.081457, -0.600066
- current_class_min_y: 2.457213
- all_maps_min_y: 2.457213

## Seed 1 / lipschitz_obstacle

Reset 51000030, t=11.

- q: 1.797633, 3.977438
- execution_action: -0.297633, -1.000000
- nominal_action: 0.449938, -0.388288
- anchor: 1.797633, 3.977438
- anchor_displacement: 0.000000, 0.000000
- correction: 0.045629, 0.059838
- proposal: 1.843262, 4.037276
- rectangle: 0
- box_low: 0.797633, 3.000000
- box_high: 2.797633, 4.000000
- emitted: 1.843262, 4.000000
- emitted_displacement: 0.045629, 0.022562
- current_class_min_y: 3.011490
- all_maps_min_y: 3.011490
