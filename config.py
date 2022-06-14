import torch

# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # To utilize only CPU even if GPU is available
# device = torch.device("cpu")

create_json = True
load_model = False
save_model = True
train_model = True
score_model = True

# Training hyperparameters
num_epochs = 1
start = 0
learning_rate = 3e-4
batch_size = 4

# File name for pth files.
filename = "models/en_de_{}.pth"