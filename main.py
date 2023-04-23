from preprocessing import preprocessor
from nltk import HiddenMarkovModelTagger
import numpy as np

def main():
    # Preprocess the data
    train_data = preprocessor.preprocess_train_data()
    test_data_dict = preprocessor.preprocess_test_data()
    gs_data_dict = preprocessor.preprocess_gs_data()

    # Prepare the tagger
    tagger = HiddenMarkovModelTagger.train(train_data)

    # Tag test data
    test_data_tagged = {}

    for key, values in zip(test_data_dict.keys(), test_data_dict.values()):
        test_data_tagged[key] = []

        for sentence in values:
            test_data_tagged[key].append(tagger.tag(sentence))
    
if __name__ == "__main__":
    main()