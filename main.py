import spacy, torch, torch.nn as nn, torch.optim as optim, torch.cuda as cuda
from tqdm import tqdm
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from torch.utils.tensorboard import SummaryWriter
from time import perf_counter
from spacy.lang.en.examples import sentences as en_sentences
from spacy.lang.de.examples import sentences as de_sentences
from config import create_json, device, load_model, save_model, train_model, num_epochs, start, learning_rate, batch_size
from model import Transformer
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint, create_json_dataset

# To install spacy languages do:
# !python -m spacy download en_core_web_sm
# !python -m spacy download en_core_web_md
# !python -m spacy download en_core_web_lg
# !python -m spacy download en_core_web_trf
# !python -m spacy download de_core_news_sm
# !python -m spacy download de_core_news_md
# !python -m spacy download de_core_news_lg
# !python -m spacy download de_dep_news_trf
# !python -m spacy download en_core_web_trf # Use this and restart the runtime
# !python -m spacy download de_dep_news_trf # Use this and restart the runtime

cuda.empty_cache()

if create_json: create_json_dataset('data/en_de/train.en', 'data/en_de/train.de')

# spacy_input = spacy.load("en_core_web_lg")
spacy_input = spacy.load("en_core_web_trf")
# spacy_output = spacy.load("de_core_news_lg")
spacy_output = spacy.load("de_dep_news_trf")

tokenize_input = lambda text: [tok.text for tok in spacy_input.tokenizer(text)]
tokenize_output = lambda text: [tok.text for tok in spacy_output.tokenizer(text)]

input_ = Field(tokenize=tokenize_input, lower=True, init_token="<start>", eos_token="<end>")
output_ = Field(tokenize=tokenize_output, lower=True, init_token="<start>", eos_token="<end>")

fields = {"English": ("eng", input_), "German": ("ger", output_)}
train_data, test_data = TabularDataset.splits(path="", train="train_en_de.json", validation="val_en_de.json", test="test_en_de.json", format="json", fields=fields)

input_.build_vocab(train_data, max_size=100_000, min_freq=2)
output_.build_vocab(train_data, max_size=100_000, min_freq=2)

# Model hyperparameters
src_vocab_size = len(input_.vocab)
trg_vocab_size = len(output_.vocab)
embedding_size = 8192
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = input_.vocab.stoi["<pad>"]

# File name for pth files.
filename = f"models/en_de_{num_epochs+start}.pth"

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=batch_size, device=device
)

model = Transformer(embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout, max_len, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

pad_idx = output_.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load(filename), model, optimizer)

sentence = en_sentences[0]

if train_model:
    s = perf_counter()
    for epoch in tqdm(range(start, start+num_epochs)):
        print(f"[Epoch {epoch+1} / {start+num_epochs}]")
        
        model.eval()
        translated_sentence = translate_sentence(
            model, sentence, input_, output_, device, max_length=50
        )

        print(f"Translated example sentence: \n {translated_sentence}")
        model.train()
        losses = []

        for batch_idx, batch in enumerate(train_iterator):
            # Get input and targets and get to cuda
            inp_data = batch.eng.to(device)
            target = batch.ger.to(device)

            # Forward prop
            output = model(inp_data, target[:-1, :])

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(output, target)
            losses.append(loss.item())

            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            # plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1

        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=filename)

    e = perf_counter()
    print(f"Training Time => {e-s:.5f} seconds")

# running on entire test data takes a while
s = perf_counter()
score = bleu(test_data[1:100], model, input_, output_, device)
print(f"Bleu score {score * 100:.2f}")
e = perf_counter()
print(f"Scoring Time => {e-s:.5f} seconds")