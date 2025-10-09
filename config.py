import torch

# General Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_HEIGHT = 320
IMG_WIDTH = 320
TEST_IMG_HEIGHT = 320
TEST_IMG_WIDTH = 320
SEED = 42

# Training Hyperparameters
LEARNING_RATE = 1e-4
NUM_EPOCHS =  1
LOG_INTERVAL = 1
BATCH_SIZE = 4

# Data Settings
DATA_DIR = "Real-World Rotational_Motion_Blur_Datasets"  # Base directory for all data
PARAMS_FILE = f"{DATA_DIR}/train/parameters_with_noise_5.csv"

# Test Data Settings
TEST_PARAMS_FILE = f"{DATA_DIR}/test/parameters_with_noise_5.csv"

# Model Hyperparameters
MODEL_NAME = "car-net" 
NUM_STAGES = 3

# Output Settings
OUTPUT_DIR = "results"
NUM_TEST_SAVE_RESULTS = 1010 # Number of test results to save
MODEL_SAVE_PATH = f"{OUTPUT_DIR}/{MODEL_NAME}_final.pth"
LOSS_PLOT_PATH = f"{OUTPUT_DIR}/loss_curve.png"
FINAL_RESULT_PATH = f"{OUTPUT_DIR}/final_comparison.png"

# Parameters for the new End-to-End RMD model
RMD_MODEL_CONFIG = {
    "num_stages": NUM_STAGES,
    "channels": 3,
    "n_blur_steps": 12,
    # Module control flags
    "use_angle_correction": True,    # Enable/disable angle correction module
    "use_refinement": True,          # Enable/disable refinement module
    "return_intermediate": True      # Return all intermediate stages or just final
}

# Parameters for the blur simulation
MIN_BLUR_ANGLE = 1.0
MAX_BLUR_ANGLE = 40.0
BLUR_N_STEPS = 12
