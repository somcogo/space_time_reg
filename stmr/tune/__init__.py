"""Hyperparameter tuning of the full pipeline with Optuna, over many patients.

Pieces (each module's docstring has the details):

    spec.py       a study = one YAML file (configs/tune/*.yaml): base config, fixed overrides,
                  search space, metric, ordered tuning cases, held-out cases, budget
    spaces.py     search spaces as define-by-run functions trial -> config overrides
    runner.py     one case = one `python -m stmr.cli` subprocess on one GPU; scored from res.pt
                  as metric(best-loss recon) - metric(zero-filled recon)
    objective.py  JournalStorage (one file, many worker processes), TPE + WilcoxonPruner,
                  the objective (cases one after another, paired pruning), the worker loop
    report.py     trials.csv / cases.csv / importance and history plots
    reeval.py     top trials + baselines on held-out cases, fresh seeds and controls
    __main__.py   CLI: create | launch | worker | status | stop | report | reeval

Typical use:

    python -m stmr.tune launch configs/tune/lambdas_v1.yaml --gpus 4,5,6,7
    python -m stmr.tune status configs/tune/lambdas_v1.yaml
    python -m stmr.tune report configs/tune/lambdas_v1.yaml
    python -m stmr.tune reeval configs/tune/lambdas_v1.yaml --gpus 4,5,6,7 --top 3 \\
        --cases holdout --seeds 0,1 --controls identity
"""
