"""cf — shared Coagulation-Fragmentation looped-reasoning code.

Extracted verbatim from CF_Milestone2_Week1_HF.ipynb (Week 1) and maze_cf.py
(Milestone 1) so the notebooks, Colab and local experiments share one source
of truth.  See README.md for the project background.

Modules
-------
cf.utils            set_seed
cf.data.curriculum  HF maze-curriculum dataset loading, ASCII adapter,
                    validation and the seeded Week-1 split.
cf.metrics          strict path metrics (no search) and their self-tests.
cf.models.rcnn      RecurrentCNN / PlainCNN baselines and n_params.
cf.train            training / evaluation utilities shared by all notebooks.
"""
