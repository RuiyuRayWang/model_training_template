import os
import pandas as pd


def load_training_log(checkpoint_dir):
    log_path = os.path.join(checkpoint_dir, 'training_log.csv')
    if not os.path.exists(log_path):
        raise FileNotFoundError(f'training_log.csv not found at: {log_path}')
    df = pd.read_csv(log_path)
    required = {
        'epoch', 'val_loss', 'f1_score', 'exact_match_acc',
        'precision', 'recall', 'train_val_ratio', 'learning_rate'
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns in training log: {sorted(missing)}')
    return df


def summarize_checkpoints(df, top_k=10):
    best = {
        'best_val_loss': df.loc[df['val_loss'].idxmin()],
        'best_f1_score': df.loc[df['f1_score'].idxmax()],
        'best_exact_match': df.loc[df['exact_match_acc'].idxmax()],
    }
    print('Best checkpoints by single metric:')
    for name, row in best.items():
        print(
            f"  {name:>16}: epoch={int(row['epoch'])}, "
            f"val_loss={row['val_loss']:.6f}, f1={row['f1_score']:.4f}, "
            f"exact={row['exact_match_acc']:.4f}, ratio={row['train_val_ratio']:.3f}"
        )

    ranked = df.copy()
    ranked['val_loss_rank'] = ranked['val_loss'].rank(method='min', ascending=True)
    ranked['f1_rank'] = ranked['f1_score'].rank(method='min', ascending=False)
    ranked['exact_rank'] = ranked['exact_match_acc'].rank(method='min', ascending=False)
    ranked['composite_rank'] = ranked[['val_loss_rank', 'f1_rank', 'exact_rank']].mean(axis=1)

    cols = [
        'epoch', 'val_loss', 'f1_score', 'exact_match_acc', 'precision',
        'recall', 'train_val_ratio', 'learning_rate', 'composite_rank'
    ]
    top = ranked.sort_values('composite_rank').head(top_k)[cols].reset_index(drop=True)
    return best, top