import pandas as pd
from sklearn.model_selection import train_test_split


def create_json_dataset(english_file: str, german_file: str, start: int=0, end: int=None, english_encoding: str="utf8", german_encoding: str="utf8"):
    if type(english_file) == str and type(german_file) == str:
        english_txt = open(english_file, encoding=english_encoding).read().split("\n")
        german_txt = open(german_file, encoding=german_encoding).read().split("\n")
    elif type(english_file) == list and type(german_file) == list:
        if len(english_file) == len(german_file):
            english_txt = []
            german_txt = []
            for eng_file, ger_file in zip(english_file, german_file):
                english_txt.extend(open(eng_file, encoding=english_encoding).read().split("\n"))
                german_txt.extend(open(ger_file, encoding=german_encoding).read().split("\n"))
        else:
            raise IndexError("Mismatch index of english_file and german_file parameters")
    else:
        raise TypeError("Invalid types for english_file and german_file parameters")
    
    df = pd.DataFrame(
        data={
            'English': [line for line in (english_txt[start:end] if end is not None else english_txt)], 
            'German': [line for line in (german_txt[start:end] if end is not None else german_txt)]
        }, 
        columns=['English', 'German']
    )
    
    train, test = train_test_split(df, test_size=0.1)
    test, val = train_test_split(test, test_size=0.5)
    
    train.to_json("train_en_de.json", orient="records", lines=True)
    test.to_json("test_en_de.json", orient="records", lines=True)
    val.to_json("val_en_de.json", orient="records", lines=True)
    del english_txt, german_txt, df, train, test, val