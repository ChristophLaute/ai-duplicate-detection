import argparse
import logging
import subprocess
import sys
from pathlib import Path


import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from runtime_config import get_device, VCDB_DIR
from utils.seeds import set_seed
from train import train, _subset, cap_negatives, load_pairs
from data.compute_negatives import construction


TRAIN_SPLIT = Path("train_pairs_segment.csv")
VAL_SPLIT = Path("val_pairs_segment.csv")
TEST_SPLIT = Path("test_pairs_segment.csv")
DEVICE = get_device()
RESULTS_DIR = Path("tune_results")
RESULTS_DIR.mkdir(exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def load_splits():
    missing = [p for p in [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing fixed split files. Run train.py once to generate:\n"
            + "\n".join(str(p) for p in missing)
        )
    train_df = pd.read_csv(TRAIN_SPLIT)
    val_df = pd.read_csv(VAL_SPLIT)
    test_df = pd.read_csv(TEST_SPLIT)
    return train_df, val_df, test_df




def recover_videoid_from_split(df):
    """Given a dataframe with 'video1' and 'video2' columns, return the set of unique video IDs."""
    new_df = set(df["video1"].unique()) | set(df["video2"].unique())
    return new_df

def run_trial(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    max_epochs: int = 60,
    patience: int = 10,
) -> float:
    result = train(
        train_df,
        val_df,
        epochs=max_epochs,
        patience=patience,
        best_metric="fbeta",
        save_model=False,
        log_csv_path=RESULTS_DIR / f"trial_{trial.number}_training_log.csv",
        train_mode=False,
        trial=trial,
    )

    return result["fbeta"]

# ---------------------------------------------------------------------------
# Separate tuning of hard_negative_ratio
# ---------------------------------------------------------------------------
#needs still work
def tune_hard_negative_ratio(
    base_lr: float,
    base_wd: float,
    base_bs: int,
    ratios: list[float] | None = None,
    max_epochs: int = 60,
    patience: int = 10,
) -> dict:
    """Grid based tuning of hard_negative_ratio with fixed other hyperparameters."""
    ratios = ratios or [0.3, 0.5, 0.7]
    base_train_df, base_val_df, _ = load_splits()
    val_ids = recover_videoid_from_split(base_val_df)
    train_ids = recover_videoid_from_split(base_train_df)
    results = {}
    for ratio in ratios:
        log.info(f"Testing hard_negative_ratio={ratio:.2f} with fixed lr={base_lr}, wd={base_wd}, bs={base_bs}")
        construction(hard_negative_ratio=ratio)
        df = load_pairs()
        train_df = _subset(df, train_ids)
        val_df = _subset(df, val_ids)
        train_df = cap_negatives(train_df, neg_pos_ratio=1.0, random_state=42)
        log.info(
            f"ratio={ratio}: train pairs={len(train_df)} "
            f"(pos={int((train_df['label'] == 1).sum())}, "
            f"neg={int((train_df['label'] == 0).sum())})"
        )
        log.info(
            f"ratio={ratio}: val pairs={len(val_df)} "
            f"(pos={int((val_df['label'] == 1).sum())}, "
            f"neg={int((val_df['label'] == 0).sum())})")
        
        result = train(
            train_df,
            val_df,
            learning_rate=base_lr,
            wd=base_wd,
            batch_size=base_bs,
            patience=patience,
            epochs=max_epochs,
            dropout=0.5,
            best_metric="fbeta",
            save_model=False,
            log_csv_path=RESULTS_DIR / f"ratio_{ratio}_training_log.csv",
            train_mode=False,
            trial=None,
        )
        results[ratio] = result["fbeta"]
        log.info(f"ratio={ratio}: val fbeta={result['fbeta']:.4f}")

            
    
        



    
    best_result = max(results.values())
    best_ratio = max(results, key=results.get)
    log.info(
        f"Best hard_negative_ratio={best_ratio} "
        f"(val fbeta={best_result:.4f})"
        )

    return results, best_result, best_ratio




def main(args):
    set_seed()
    if args.tune_ratio:
        log.info("Starting separate tuning of hard_negative_ratio...")
        tune_hard_negative_ratio(
            base_lr=args.base_lr or 1e-4,
            base_wd=args.base_wd or 1e-4,
            base_bs=args.base_bs or 64,
            ratios=args.ratios,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        return
    log.info("Loading pairs from disk...")
    train_df, val_df, _test_df = load_splits()

    log.info(
        f"Train pairs: {len(train_df)} "
        f"(pos={int((train_df['label']==1).sum())}, neg={int((train_df['label']==0).sum())})"
    )
    log.info(
        f"Val pairs:   {len(val_df)} "
        f"(pos={int((val_df['label']==1).sum())}, neg={int((val_df['label']==0).sum())})"
    )

    storage = f"sqlite:///{RESULTS_DIR / 'optuna_study.db'}"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        storage=storage,
        load_if_exists=True,
    )

    log.info(
        f"Study '{args.study_name}': {len(study.trials)} existing trials. "
        f"Running {args.n_trials} more."
    )

    def objective(trial):
        return run_trial(
            trial,
            train_df,
            val_df,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[_log_callback],
    )

    best = study.best_trial
    log.info("=" * 60)
    log.info(f"Best trial #{best.number}  val fbeta = {best.value:.4f}")
    for k, v in best.params.items():
        log.info(f"  {k}: {v}")
    log.info("=" * 60)

    # Save summary
    summary_path = RESULTS_DIR / "best_params.txt"
    with open(summary_path, "w") as f:
        f.write(f"Best trial: #{best.number}\n")
        f.write(f"Val fbeta:     {best.value:.4f}\n\n")
        for k, v in best.params.items():
            f.write(f"{k}: {v}\n")
    log.info(f"Saved best params to {summary_path}")

    df_trials = study.trials_dataframe()
    df_trials.to_csv(RESULTS_DIR / "all_trials.csv", index=False)
    log.info(f"All trial results saved to {RESULTS_DIR / 'all_trials.csv'}")


def _log_callback(study, trial):
    if trial.state == optuna.trial.TrialState.PRUNED:
        log.info(f"Trial #{trial.number} pruned.")
    elif trial.state == optuna.trial.TrialState.COMPLETE:
        log.info(
            f"Trial #{trial.number} done. Val fbeta={trial.value:.4f} | "
            f"Best so far: {study.best_value:.4f} | Params: {trial.params}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for Siamese network")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials to run")
    parser.add_argument("--max-epochs", type=int, default=60, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience per trial")
    parser.add_argument("--study-name", type=str, default="siamese_tuning", help="Optuna study name (use same name to resume)")
    parser.add_argument("--tune-ratio", action="store_true", help="Run separate tuning of hard_negative_ratio with fixed other hyperparameters")
    parser.add_argument("--base-lr", type=float, default=None)
    parser.add_argument("--base-wd", type=float, default=None)
    parser.add_argument("--base-bs", type=int, default=None)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7], help="List of hard_negative_ratios to test (e.g. --ratios 0.3 0.5 0.7)")

    args = parser.parse_args()
    main(args)
