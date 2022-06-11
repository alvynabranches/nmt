import spacy, torch, torch.nn as nn, torch.optim as optim, torch.cuda as cuda
from tqdm import tqdm
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from torch.utils.tensorboard import SummaryWriter
from time import perf_counter
from spacy.lang.en.examples import sentences as en_sentences
from spacy.lang.de.examples import sentences as de_sentences
from config import create_json, device, load_model, save_model, score_model, train_model, num_epochs, start, learning_rate, batch_size
from model import Transformer
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint, create_json_dataset

cuda.empty_cache()

if create_json: create_json_dataset('data/en_de/train.en', 'data/en_de/train.de')

spacy_input = spacy.load("en_core_web_sm")
# spacy_input = spacy.load("en_core_web_trf")
spacy_output = spacy.load("de_core_news_sm")
# spacy_output = spacy.load("de_dep_news_trf")

input_ = Field(tokenize=lambda text: [tok.text for tok in spacy_input.tokenizer(text)], lower=True, init_token="<sos>", eos_token="<eos>")
output_ = Field(tokenize=lambda text: [tok.text for tok in spacy_output.tokenizer(text)], lower=True, init_token="<sos>", eos_token="<eos>")

train_data, val_data, test_data = TabularDataset.splits(path="", train="train_en_de.json", validation="val_en_de.json", test="test_en_de.json", format="json", fields={"English": ("src", input_), "German": ("trg", output_)})

input_.build_vocab(train_data, max_size=10_000, min_freq=2)
output_.build_vocab(train_data, max_size=10_000, min_freq=2)

# Model hyperparameters
src_vocab_size = len(input_.vocab)
trg_vocab_size = len(output_.vocab)
embedding_size = 4096
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
step = 0 + start

train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train_data, val_data, test_data), batch_size=batch_size, sort_within_batch=True, sort_key=lambda x: len(x.src), device=device
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
    for epoch in range(start, start+num_epochs):
        print(f"[Epoch {epoch+1} / {start+num_epochs}]")
        
        model.train()
        losses = []

        for batch_idx, batch in tqdm(enumerate(train_iterator)):
            # Get input and targets and get to cuda
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

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

        print()
        es = perf_counter()
        model.eval()
        translated_sentence = translate_sentence(
            model, sentence, input_, output_, device, max_length=max_len
        )
        print(f"Translated example sentence: \n {translated_sentence}")
        ee = perf_counter()
        print(f"Evaluating Time => {ee-es:.5f} seconds")

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

if score_model:
    # running on entire test data takes a while
    s = perf_counter()
    score = bleu(test_data[:100], model, input_, output_, device)
    print(f"Bleu score {score * 100:.2f}")
    e = perf_counter()
    print(f"Scoring Time => {e-s:.5f} seconds")