from preprocessing import preprocessor
from evaluation import evaluator
from nltk import HiddenMarkovModelTagger

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

    metrics_dict = evaluator.evaluate(test_data_tagged, gs_data_dict)

    # Compound metrics to evaluate whole tagger
    TP = 0; FP = 0; FN = 0
    for tag, metrics in zip(metrics_dict.keys(), metrics_dict.values()):
        evaluator.generate_confusion_matrix(tag, metrics['TP'], metrics['FP'], metrics['FN'])
        TP += metrics['TP']; FP += metrics['FP']; FN += metrics['FN']

    # Evaluate tagger
    evaluator.generate_confusion_matrix('TAGGER', TP, FP, FN)
    
    return
    
if __name__ == "__main__":
    main()