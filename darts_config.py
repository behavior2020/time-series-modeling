import torch
from darts.metrics import mae
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# File name
file_name = "PLEASE_ENTER_YOUR_FILE_PATH_HERE.xlsx"  # Update this to your file path

# Global variables
ZEROS_MAX_PERCENT = 0.95  # remove rows with n percent of zero values
SEASONALITY = 12  # monthly periods
DAYS_PER_MONTH = 30.44  # average days per month for Prophet
FOURIER_ORDER = 5  # number of Fourier orders for Prophet

# Model variables
NUM_EPOCHS = 50  # set number of epochs for training deep learning models
INPUT_LENGTH = 24
OUTPUT_LENGTH = 6

# Backtesting variables
VAL_WINDOW = 5
FORECAST_PERIODS = 6  # originally 3
TRAIN_PERCENTAGE = 0.85
TRAIN_N_POINTS = 26

# # Exponential Smoothing vs. N-HiTS
METRIC = mae
METRIC_NAME = "MAE"

# Model parameters
optimizer_kwargs = {
    "lr": 1e-3,
}

# PyTorch Lightning Trainer arguments
pl_trainer_kwargs = {
    "gradient_clip_val": 1,  # prevent exploding gradient
    "max_epochs": 200,  # max number of complete pases of the training dataset through the algorithm
    "accelerator": "auto",  # recognizes the machine you are on, and selects the appropriate Accelerator
    "callbacks": [],
}

# Learning rate scheduler
lr_scheduler_cls = (
    torch.optim.lr_scheduler.ExponentialLR
)  # Decays the learning rate of each parameter group by gamma every epoch. When last_epoch=-1, sets initial lr as lr.
lr_scheduler_kwargs = {
    "gamma": 0.999,  # Multiplicative factor of learning rate decay
}

# Early stopping (needs to be reset for each model later on)
# this setting stops training once the the validation loss has not decreased by more than 1e-3 for 10 epochs
early_stopping_args = {
    "monitor": "train_loss",  # no validation set (val_loss) because backtesting does not have a validation set
    "patience": 10,  # Number of checks without improvement after which training will be stopped
    "min_delta": 1e-3,  #  minimum change in the monitored quantity to qualify as an improvement
    "mode": "min",  # minimize the monitored metric
}

# Common model arguments
common_model_args = {
    "input_chunk_length": INPUT_LENGTH,  # lookback window
    "output_chunk_length": OUTPUT_LENGTH,  # forecast/lookahead window
    "optimizer_kwargs": optimizer_kwargs,
    "pl_trainer_kwargs": pl_trainer_kwargs,
    "lr_scheduler_cls": lr_scheduler_cls,
    "lr_scheduler_kwargs": lr_scheduler_kwargs,
    "likelihood": None,  # use a likelihood for probabilistic forecasts
    "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
    "force_reset": True,  # If set to True, any previously-existing model with the same name will be reset (all checkpoints will be discarded)
    # "batch_size":
    "random_state": 42,
}

# Establish callbacks for deep learning models
pl_trainer_kwargs["callbacks"] = [EarlyStopping(**early_stopping_args)]  # callbacks need to be passed as a list
