import itertools
import random, torch
from my_dataloader import get_data 
from models import ResNet18Regression, LinearRegression, EfficientNetRegression, MobileNetRegression, SqueezeNetRegression, TinyCNNRegression
from model_utility_functions import train_and_validate_model_k
import numpy as np

USE_GPU = True
#MODEL = "squeeze" #resnet
models = ["squeeze","mobile","resnet"]
LOSS = "mse_cnn"
TEST_SETS = 10
DEV_SETS = 10
NUM_EPOCHS = 32
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4 
DATASET = "cage"
image_folder = "mel_spectrograms_128"
log_file = "logs/folds.txt"

NUM_CLASSES = 2
NUM_BINS = 128

def log_fold(test, dev,
             dev_mae, test_mae,
             dev_rmse, test_rmse,
             dev_r2, test_r2,
             file_path):
    with open(file_path, "a") as file:
        file.write(
            f"{test}, {dev}, "
            f"{dev_mae:.6f}, {test_mae:.6f}, "
            f"{dev_rmse:.6f}, {test_rmse:.6f}, "
            f"{dev_r2:.6f}, {test_r2:.6f}\n"
        )

def main():
    torch.cuda.empty_cache()
    dev_acc = 0
    test_acc = 0
    dev_auc = 0
    test_auc = 0

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For GPU determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open(log_file, "w") as f:
        f.write("test_fold, dev_fold, dev_MAE, test_MAE, dev_RMSE, test_RMSE, dev_R2, test_R2\n")

    if USE_GPU:
        processor = torch.device("cuda")
    else:
        processor = torch.device("cpu")
        
    for MODEL in models:
        for test_fold in range(0,TEST_SETS):
            for dev_fold in range(0,DEV_SETS):
                if test_fold != dev_fold:
                    train_set, dev_set, test_set = get_data(dataset=DATASET,data_folds='data/'+DATASET+'/stratified_folds',test_fold=test_fold,dev_fold=dev_fold,image_folder="data/"+DATASET+'/'+image_folder,loss=LOSS,batch_size=BATCH_SIZE,num_outer_folds=10)
                    
                    if MODEL == "lr":
                        model = LinearRegression(NUM_BINS).to(processor)
                            
                    elif MODEL == "tiny":
                        model = TinyCNNRegression().to(processor)

                    elif MODEL == "squeeze":
                        model = SqueezeNetRegression().to(processor)

                    elif MODEL == "mobile":
                        model = MobileNetRegression().to(processor)

                    elif MODEL == "resnet":
                        model = ResNet18Regression().to(processor)

                    elif MODEL == "efficient":
                        model = EfficientNetRegression().to(processor)

                    (dev_mae, dev_rmse, dev_r2, dev_avg_loss,
                    test_mae, test_rmse, test_r2, test_avg_loss) = train_and_validate_model_k(model, train_set, dev_set, test_set,dev_fold,test_fold, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, processor)
                    
                    log_fold(test_fold, dev_fold,
                            dev_mae, test_mae,
                            dev_rmse, test_rmse,
                            dev_r2, test_r2,
                            log_file)

if __name__ == "__main__":
    main()