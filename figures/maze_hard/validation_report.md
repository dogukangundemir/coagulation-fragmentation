# Maze-hard 30x30 validation report

Validation table: number of failing rows per check (1000 train / 1000 test grids; 0 = valid).

| check | train | test |
|---|---:|---:|
| length900 | 0 | 0 |
| charset | 0 | 0 |
| no_o_in_question | 0 | 0 |
| one_S_one_G | 0 | 0 |
| walls_match | 0 | 0 |
| endpoints_match | 0 | 0 |
| path_on_free | 0 | 0 |
| single_simple_path | 0 | 0 |
| rating_present | 0 | 0 |

Duplicates: within_train=0 within_test=0 across_splits=0

## train (n=1000)

- rating: min=110 mean=115.06 max=147
- path length: min=110 mean=115.06 max=147 cells
- rating == number of label cells (S+G+o): 1000/1000
- shortest S->G paths per maze (BFS count, capped at 3): 3 path(s): 1000 mazes
- mazes with more than one shortest path: 1000/1000

## test (n=1000)

- rating: min=110 mean=114.52 max=154
- path length: min=110 mean=114.52 max=154 cells
- rating == number of label cells (S+G+o): 1000/1000
- shortest S->G paths per maze (BFS count, capped at 3): 3 path(s): 1000 mazes
- mazes with more than one shortest path: 1000/1000

