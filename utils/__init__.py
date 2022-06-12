import pandas as pd, spacy, torch
from torchtext.data.metrics import bleu_score
from sklearn.model_selection import train_test_split


def translate_sentence(model, sentence, input_, english, device, max_length=50, input_vocab="de_core_news_md"):
    # Load input tokenizer
    spacy_ger = spacy.load(input_vocab)

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, input_.init_token)
    tokens.append(input_.eos_token)

    # Go through each input token and convert to an index
    text_to_indices = [input_.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, input_, output_, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["eng"]
        trg = vars(example)["ger"]

        prediction = translate_sentence(model, src, input_, output_, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(model, optimizer, filename="new_checkpoint.pth"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print("=>Saved checkpoint")


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    

def create_json_dataset(english_file: str, german_file: str, start: int=0, end: int=None, english_encoding: str="utf8", german_encoding: str="utf8"):
    english_txt = open(english_file, encoding=english_encoding).read().split("\n")
    german_txt = open(german_file, encoding=german_encoding).read().split("\n")
    
    df = pd.DataFrame(
        data={
            'English': [line for line in (english_txt[0:end] if end is not None else english_txt)], 
            'German': [line for line in (german_txt[0:end] if end is not None else german_txt)]
        }, 
        columns=['English', 'German']
    )
    
    train, test = train_test_split(df, test_size=0.2)
    test, val = train_test_split(test, test_size=0.5)
    
    train.to_json("train_en_de.json", orient="records", lines=True)
    test.to_json("test_en_de.json", orient="records", lines=True)
    val.to_json("val_en_de.json", orient="records", lines=True)
    del english_txt, german_txt, df, train, test, val