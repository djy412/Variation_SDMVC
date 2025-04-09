# Parameter configurations
BATCH_SIZE = 256
PRE_TRAIN_EPOCHS = 500
FINE_TUNE_EPOCHS = 200
MODEL_FILENAME = './data/model.ckpt'
# Learning rate for both encoder and decoder
LR = 0.001
LAMBDA = 0.1 #--- How much reconstruction loss to include in Triplet loss
GAMMA = 0.1 #--- How much clustering factor to include in the Clutering loss
LATENT_DIM = 10
NUM_CLASSES = 4
NUM_VIEWS = 2
TOLERANCE = 0.001 #--- How close to the last estimage is good enough
PATIENCE = 5 #--- How many iterations need to be within the tolerance
UPDATE_INTERVAL = 5 #--- How often to update the estimated "true data", 1 would = updating every Epoch
# Load data in parallel by choosing the best num of workers for your system
WORKERS = 8
#dataset_name = 'MULTI-MNIST'
#dataset_name = 'MULTI-FASHION'
#dataset_name = 'FASHION-MV'
#dataset_name = 'MULTI-MVP-N'
#dataset_name = 'MULTI_STL-10'
dataset_name = 'MULTI_Eglin'
#dataset_name = 'MULTI_2V_Market'
CHANNELS = 3
IHMC = False