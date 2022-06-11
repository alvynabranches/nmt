import torch

create_json = True

# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # To utilize only CPU even if GPU is available
# device = torch.device("cpu")

load_model = False
save_model = True
train_model = True
score_model = True

# Training hyperparameters
num_epochs = 100
start = 0
learning_rate = 3e-4
batch_size = 32