# (a) Where the critic's positive-goal law puts its mass along the existing futures

Law: geometric gamma 0.999 truncated at the branch end (the critic stream's own draw); anchor-weighted means of per-branch masses.  reach = within 0.5 of the task goal; goal_area = the maze region; stall = |delta state| < 0.001; death_pos = last 5 rows of a death branch; terminal = last 10 rows; mid_route = neither reach nor stall nor death_pos.  `goal_mass.json`.

| source | n | weight | mean length | success / death / timeout | reach 0.5 | goal_area | stall | death_pos | terminal 10 | zones | far legs | mid_route |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| O recorded | all anchors | 53747 | 1.0000 | 135 | 0.999 / 0.000 / 0.001 | 0.0224 | 0.2731 | 0.022 | 0.000 | 0.155 | 0.159 | 0.044 | 0.955 |
| O recorded | start anchors | 4276 | 0.0794 | 255 | 1.000 / 0.000 / 0.000 | 0.0035 | 0.0763 | 0.030 | 0.000 | 0.036 | 0.194 | 0.042 | 0.966 |
| CF sealed (start-agent continuation) | all anchors | 53747 | 1.0000 | 143 | 0.621 / 0.291 / 0.088 | 0.0194 | 0.2730 | 0.051 | 0.056 | 0.230 | 0.161 | 0.040 | 0.873 |
| CF sealed | start anchors | 4276 | 0.0794 | 138 | 0.243 / 0.728 / 0.029 | 0.0009 | 0.0377 | 0.015 | 0.053 | 0.115 | 0.144 | 0.028 | 0.931 |
| lineage 0 queries: logged | start anchors | 8552 | 0.1589 | 209 | 0.180 / 0.692 / 0.128 | 0.0006 | 0.0164 | 0.076 | 0.050 | 0.107 | 0.155 | 0.041 | 0.874 |
| lineage 0 queries: mode | start anchors | 8552 | 0.1589 | 224 | 0.195 / 0.667 / 0.138 | 0.0006 | 0.0165 | 0.077 | 0.048 | 0.104 | 0.147 | 0.067 | 0.874 |
| lineage 0 queries: samples | start anchors | 34208 | 0.6355 | 221 | 0.195 / 0.670 / 0.135 | 0.0006 | 0.0169 | 0.076 | 0.048 | 0.105 | 0.147 | 0.063 | 0.875 |
| lineage 0 queries: added | start anchors | 42760 | 0.7943 | 222 | 0.195 / 0.669 / 0.136 | 0.0006 | 0.0169 | 0.076 | 0.048 | 0.104 | 0.147 | 0.064 | 0.875 |
| lineage 0 queries: extended | start anchors | 51312 | 0.9532 | 220 | 0.193 / 0.673 / 0.134 | 0.0006 | 0.0168 | 0.076 | 0.049 | 0.105 | 0.149 | 0.060 | 0.875 |
| lineage 1 queries: logged | start anchors | 8552 | 0.1589 | 231 | 0.185 / 0.653 / 0.162 | 0.0006 | 0.0167 | 0.115 | 0.048 | 0.104 | 0.161 | 0.043 | 0.837 |
| lineage 1 queries: mode | start anchors | 8552 | 0.1589 | 236 | 0.208 / 0.638 / 0.155 | 0.0007 | 0.0174 | 0.108 | 0.047 | 0.102 | 0.156 | 0.066 | 0.845 |
| lineage 1 queries: samples | start anchors | 34208 | 0.6355 | 239 | 0.200 / 0.638 / 0.162 | 0.0007 | 0.0168 | 0.113 | 0.047 | 0.102 | 0.156 | 0.062 | 0.839 |
| lineage 1 queries: added | start anchors | 42760 | 0.7943 | 238 | 0.202 / 0.638 / 0.160 | 0.0007 | 0.0169 | 0.112 | 0.047 | 0.102 | 0.156 | 0.063 | 0.840 |
| lineage 1 queries: extended | start anchors | 51312 | 0.9532 | 237 | 0.199 / 0.640 / 0.161 | 0.0007 | 0.0169 | 0.113 | 0.047 | 0.102 | 0.157 | 0.059 | 0.840 |
| lineage 2 queries: logged | start anchors | 8552 | 0.1589 | 202 | 0.220 / 0.678 / 0.102 | 0.0007 | 0.0190 | 0.063 | 0.046 | 0.100 | 0.153 | 0.039 | 0.891 |
| lineage 2 queries: mode | start anchors | 8552 | 0.1589 | 209 | 0.235 / 0.663 / 0.102 | 0.0007 | 0.0192 | 0.061 | 0.045 | 0.098 | 0.150 | 0.053 | 0.893 |
| lineage 2 queries: samples | start anchors | 34208 | 0.6355 | 207 | 0.230 / 0.668 / 0.102 | 0.0007 | 0.0188 | 0.061 | 0.045 | 0.099 | 0.150 | 0.051 | 0.894 |
| lineage 2 queries: added | start anchors | 42760 | 0.7943 | 207 | 0.231 / 0.667 / 0.102 | 0.0007 | 0.0189 | 0.061 | 0.045 | 0.099 | 0.150 | 0.052 | 0.894 |
| lineage 2 queries: extended | start anchors | 51312 | 0.9532 | 206 | 0.229 / 0.669 / 0.102 | 0.0007 | 0.0189 | 0.061 | 0.045 | 0.099 | 0.151 | 0.050 | 0.893 |

## By outcome (per source: mean length; reach / goal_area / stall / death_pos / terminal masses)

| source | outcome | n | mean length | reach 0.5 | goal_area | stall | death_pos | terminal 10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| O recorded | all anchors | success | 53688 | 135 | 0.0224 | 0.2724 | 0.022 | 0.000 | 0.155 |
| O recorded | all anchors | timeout | 59 | 410 | 0.0000 | 0.9355 | 0.236 | 0.000 | 0.049 |
| O recorded | start anchors | success | 4276 | 255 | 0.0035 | 0.0763 | 0.030 | 0.000 | 0.036 |
| CF sealed (start-agent continuation) | all anchors | success | 33402 | 111 | 0.0313 | 0.3980 | 0.001 | 0.000 | 0.202 |
| CF sealed (start-agent continuation) | all anchors | death | 15612 | 50 | 0.0000 | 0.0000 | 0.001 | 0.193 | 0.354 |
| CF sealed (start-agent continuation) | all anchors | timeout | 4733 | 678 | 0.0000 | 0.2919 | 0.575 | 0.000 | 0.011 |
| CF sealed | start anchors | success | 1040 | 237 | 0.0039 | 0.1130 | 0.000 | 0.000 | 0.039 |
| CF sealed | start anchors | death | 3115 | 80 | 0.0000 | 0.0000 | 0.000 | 0.072 | 0.145 |
| CF sealed | start anchors | timeout | 121 | 781 | 0.0000 | 0.3583 | 0.532 | 0.000 | 0.009 |
| lineage 0 queries: logged | start anchors | success | 1565 | 283 | 0.0034 | 0.0810 | 0.000 | 0.000 | 0.034 |
| lineage 0 queries: logged | start anchors | death | 5912 | 83 | 0.0000 | 0.0000 | 0.000 | 0.072 | 0.144 |
| lineage 0 queries: logged | start anchors | timeout | 1075 | 790 | 0.0000 | 0.0139 | 0.590 | 0.000 | 0.008 |
| lineage 0 queries: mode | start anchors | success | 1677 | 308 | 0.0032 | 0.0773 | 0.000 | 0.000 | 0.032 |
| lineage 0 queries: mode | start anchors | death | 5707 | 82 | 0.0000 | 0.0000 | 0.000 | 0.072 | 0.145 |
| lineage 0 queries: mode | start anchors | timeout | 1168 | 790 | 0.0000 | 0.0105 | 0.555 | 0.000 | 0.008 |
| lineage 0 queries: samples | start anchors | success | 6721 | 306 | 0.0032 | 0.0775 | 0.000 | 0.000 | 0.032 |
| lineage 0 queries: samples | start anchors | death | 22906 | 82 | 0.0000 | 0.0000 | 0.000 | 0.072 | 0.145 |
| lineage 0 queries: samples | start anchors | timeout | 4581 | 790 | 0.0000 | 0.0134 | 0.565 | 0.000 | 0.008 |
| lineage 0 queries: added | start anchors | success | 8398 | 307 | 0.0032 | 0.0774 | 0.000 | 0.000 | 0.032 |
| lineage 0 queries: added | start anchors | death | 28613 | 82 | 0.0000 | 0.0000 | 0.000 | 0.072 | 0.145 |
| lineage 0 queries: added | start anchors | timeout | 5749 | 790 | 0.0000 | 0.0128 | 0.563 | 0.000 | 0.008 |
| lineage 0 queries: extended | start anchors | success | 9963 | 303 | 0.0032 | 0.0780 | 0.000 | 0.000 | 0.032 |
| lineage 0 queries: extended | start anchors | death | 34525 | 82 | 0.0000 | 0.0000 | 0.000 | 0.072 | 0.145 |
| lineage 0 queries: extended | start anchors | timeout | 6824 | 790 | 0.0000 | 0.0130 | 0.567 | 0.000 | 0.008 |
| lineage 1 queries: logged | start anchors | success | 1607 | 279 | 0.0034 | 0.0815 | 0.000 | 0.000 | 0.034 |
| lineage 1 queries: logged | start anchors | death | 5563 | 79 | 0.0000 | 0.0000 | 0.000 | 0.073 | 0.147 |
| lineage 1 queries: logged | start anchors | timeout | 1382 | 790 | 0.0000 | 0.0096 | 0.709 | 0.000 | 0.008 |
| lineage 1 queries: mode | start anchors | success | 1775 | 306 | 0.0032 | 0.0775 | 0.000 | 0.000 | 0.032 |
| lineage 1 queries: mode | start anchors | death | 5448 | 79 | 0.0000 | 0.0000 | 0.000 | 0.074 | 0.147 |
| lineage 1 queries: mode | start anchors | timeout | 1329 | 790 | 0.0000 | 0.0084 | 0.696 | 0.000 | 0.008 |
| lineage 1 queries: samples | start anchors | success | 6905 | 303 | 0.0032 | 0.0782 | 0.000 | 0.000 | 0.033 |
| lineage 1 queries: samples | start anchors | death | 21810 | 79 | 0.0000 | 0.0000 | 0.000 | 0.074 | 0.148 |
| lineage 1 queries: samples | start anchors | timeout | 5493 | 790 | 0.0000 | 0.0073 | 0.700 | 0.000 | 0.008 |
| lineage 1 queries: added | start anchors | success | 8680 | 303 | 0.0032 | 0.0780 | 0.000 | 0.000 | 0.033 |
| lineage 1 queries: added | start anchors | death | 27258 | 79 | 0.0000 | 0.0000 | 0.000 | 0.074 | 0.148 |
| lineage 1 queries: added | start anchors | timeout | 6822 | 790 | 0.0000 | 0.0075 | 0.699 | 0.000 | 0.008 |
| lineage 1 queries: extended | start anchors | success | 10287 | 299 | 0.0033 | 0.0786 | 0.000 | 0.000 | 0.033 |
| lineage 1 queries: extended | start anchors | death | 32821 | 79 | 0.0000 | 0.0000 | 0.000 | 0.074 | 0.147 |
| lineage 1 queries: extended | start anchors | timeout | 8204 | 790 | 0.0000 | 0.0079 | 0.701 | 0.000 | 0.008 |
| lineage 2 queries: logged | start anchors | success | 1880 | 283 | 0.0033 | 0.0764 | 0.000 | 0.000 | 0.033 |
| lineage 2 queries: logged | start anchors | death | 5800 | 87 | 0.0000 | 0.0000 | 0.000 | 0.068 | 0.136 |
| lineage 2 queries: logged | start anchors | timeout | 872 | 790 | 0.0000 | 0.0220 | 0.614 | 0.000 | 0.008 |
| lineage 2 queries: mode | start anchors | success | 2014 | 298 | 0.0032 | 0.0732 | 0.000 | 0.000 | 0.032 |
| lineage 2 queries: mode | start anchors | death | 5670 | 87 | 0.0000 | 0.0000 | 0.000 | 0.068 | 0.136 |
| lineage 2 queries: mode | start anchors | timeout | 868 | 790 | 0.0000 | 0.0198 | 0.600 | 0.000 | 0.008 |
| lineage 2 queries: samples | start anchors | success | 7929 | 295 | 0.0032 | 0.0742 | 0.000 | 0.000 | 0.032 |
| lineage 2 queries: samples | start anchors | death | 22830 | 88 | 0.0000 | 0.0000 | 0.000 | 0.068 | 0.136 |
| lineage 2 queries: samples | start anchors | timeout | 3449 | 789 | 0.0000 | 0.0174 | 0.594 | 0.000 | 0.008 |
| lineage 2 queries: added | start anchors | success | 9943 | 296 | 0.0032 | 0.0740 | 0.000 | 0.000 | 0.032 |
| lineage 2 queries: added | start anchors | death | 28500 | 88 | 0.0000 | 0.0000 | 0.000 | 0.068 | 0.136 |
| lineage 2 queries: added | start anchors | timeout | 4317 | 790 | 0.0000 | 0.0179 | 0.595 | 0.000 | 0.008 |
| lineage 2 queries: extended | start anchors | success | 11823 | 294 | 0.0032 | 0.0744 | 0.000 | 0.000 | 0.032 |
| lineage 2 queries: extended | start anchors | death | 34300 | 87 | 0.0000 | 0.0000 | 0.000 | 0.068 | 0.136 |
| lineage 2 queries: extended | start anchors | timeout | 5189 | 790 | 0.0000 | 0.0186 | 0.598 | 0.000 | 0.008 |
