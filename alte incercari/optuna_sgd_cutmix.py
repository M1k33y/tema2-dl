# Run AFTER notebook cells 0-6 (imports, data, VGG13), INSTEAD of cells 8-14.
# Install Optuna first if needed: %pip install -q optuna
import os, time, math, random, gc, json
import numpy as np
import torch
import optuna
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2

SEARCH_HOURS = 4.0  # 3.0, 4.0, or 5.0
TRIAL_EPOCHS = 55  # screening, not final model
MAX_TRIALS = 30
SEED = 404
NUM_CLASSES = 100
WORKDIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
STUDY_NAME = 'vgg13_sgd_cutmix_55ep_v1'
DB_PATH = os.path.join(WORKDIR, 'optuna_sgd_cutmix_55ep.db')
CSV_PATH = os.path.join(WORKDIR, 'optuna_sgd_cutmix_55ep.csv')
BEST_JSON = os.path.join(WORKDIR, 'optuna_sgd_cutmix_best.json')
# Reserve time for finishing an ongoing trial; Optuna's timeout does not interrupt a trial.
SEARCH_DEADLINE = time.monotonic() + SEARCH_HOURS * 3600

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# The existing notebook uses batch_size=128. Create dedicated batch=64 loaders.
search_train_loader = DataLoader(
    train_set, batch_size=64, shuffle=True, num_workers=2,
    pin_memory=(device.type == 'cuda'), persistent_workers=True
)
search_val_loader = DataLoader(
    val_set, batch_size=128, shuffle=False, num_workers=2,
    pin_memory=(device.type == 'cuda'), persistent_workers=True
)

def objective(trial):
    seed_everything(SEED)  # same seed for comparable trials
    lr = trial.suggest_float('lr', 0.03, 0.07)
    wd = trial.suggest_float('weight_decay', 1e-4, 1e-3, log=True)
    smoothing = trial.suggest_float('label_smoothing', 0.03, 0.14)
    cutmix_prob = trial.suggest_float('cutmix_prob', 0.15, 0.45)
    cutmix_alpha = trial.suggest_float('cutmix_alpha', 0.4, 1.4)
    model = optimizer = scheduler = scaler = None
    try:
        model = torch.jit.script(VGG13().to(device))
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd, nesterov=True,
                              fused=(device.type == 'cuda'))
        warmup = 5
        min_lr = 5e-4
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = min(1., (epoch - warmup) / max(1, TRIAL_EPOCHS - warmup))
            return (min_lr / lr) + (1 - min_lr / lr) * 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
        cutmix = v2.CutMix(num_classes=NUM_CLASSES, alpha=cutmix_alpha)
        best = 0.
        for epoch in range(TRIAL_EPOCHS):
            # Avoid beginning another epoch if too close to search deadline.
            if time.monotonic() + 75 > SEARCH_DEADLINE:
                raise optuna.TrialPruned('Search time budget reached')
            model.train()
            for x, y in search_train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if random.random() < cutmix_prob:
                    x, y = cutmix(x, y)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            model.eval()
            correct = torch.zeros((), dtype=torch.long, device=device)
            count = 0
            with torch.inference_mode():
                for x, y in search_val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                        logits = model(x)
                    correct += (logits.argmax(1) == y).sum()
                    count += y.size(0)
            acc = 100 * correct.item() / count
            best = max(best, acc)
            trial.report(acc, epoch)
            scheduler.step()
            if epoch >= 29 and trial.should_prune():
                raise optuna.TrialPruned(f'epoch={epoch + 1}, accuracy={acc:.2f}%')
        return best
    finally:
        del model, optimizer, scheduler, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

study = optuna.create_study(
    study_name=STUDY_NAME,
    storage='sqlite:///' + DB_PATH,
    load_if_exists=True,
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=5)
)

def on_trial_end(study, trial):
    study.trials_dataframe().to_csv(CSV_PATH, index=False)
    if study.best_trials:
        best = study.best_trial
        with open(BEST_JSON, 'w') as f:
            json.dump({'validation_no_tta': best.value, 'params': best.params,
                       'trial_number': best.number, 'screening_epochs': TRIAL_EPOCHS}, f, indent=2)
        print(f'Trial {trial.number}: {trial.state.name}; best={best.value:.2f}% (no TTA)')

print(f'Optuna: max {SEARCH_HOURS:.1f} hours, {TRIAL_EPOCHS} epochs/trial, batch=64')
study.optimize(objective, n_trials=MAX_TRIALS,
               timeout=max(0, SEARCH_DEADLINE - time.monotonic()),
               callbacks=[on_trial_end], n_jobs=1)
if study.best_trials:
    print('BEST SCREENING ACCURACY (NO TTA):', study.best_value)
    print('BEST PARAMS:', study.best_params)
    print('Results:', CSV_PATH)
    print('Config:', BEST_JSON)
else:
    print('No completed trial. Consider increasing SEARCH_HOURS.')
