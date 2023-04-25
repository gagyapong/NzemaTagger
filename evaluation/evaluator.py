import matplotlib.pyplot as plt
import os

metrics_dict = {}
taglist = ['PRON', 'SCONJ', 'ADV', 'DET', 'INTJ', 'X', 'PART', 'ADP',
                'SYM', 'NUM', 'CCONJ', 'VERB', 'PROPN', 'ADJ', 'NOUN']

def create_metrics_dict():
    '''
    Creates dictionary with metrics (TP | FP | FN) for each tag in taglist
    Note that, the nature of the model does not allow for true negatives (TN)
    '''
    
    for tag in taglist:
        metrics_dict[tag] = {
            'TP': 0,
            'FP': 0,
            'FN': 0
        }

def evaluate(test_dict, gs_dict):
    '''
    Takes in the test and golden data dictionaries and evaluates the performance of the tagger
    Returns a metrics dictionary for each tag in the taglist
    '''

    # Initialize metrics dictionary
    create_metrics_dict()

    # Iterate through passages
    for passage in test_dict.keys():
        for test_sent, gs_sent in zip(test_dict[passage], gs_dict[passage]):
            for test, gs in zip(test_sent, gs_sent):
                if test == gs:
                    metrics_dict[gs[1]]['TP'] += 1

                else:
                    metrics_dict[test[1]]['FP'] += 1
                    metrics_dict[gs[1]]['FN'] += 1    
                    
    return metrics_dict

def generate_confusion_matrix(tag, TP, FP, FN):
    '''
    Generates confusion matrices for visualization purposes
    Code generated with ChatGPT
    '''

    TN = 0 # For the sake of completeness
    
    # Create confusion matrix plot
    confusion_matrix_plot = plt.figure(figsize=(4, 4))
    plt.imshow([[TP, FP], [FN, TN]], cmap='OrRd')

    # Set plot properties
    plt.xticks([0, 1], ['Positive', 'Negative'], fontsize=12)
    plt.yticks([0, 1], ['Positive', 'Negative'], fontsize=12)
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{[[TP, FP], [FN, TN]][i][j]:.0f}', ha='center', va='center', color='white')

    accuracy, precision, recall, f1 = calc_metrics(TP, FP, FN)

    # Add metrics as text annotations
    plt.text(-1, -1.1, f'{tag}\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}',
         ha='left', va='center', fontsize=12, fontweight='bold')

    # If path doesn't already exist, create it
    if not os.path.exists('./data/plots'):
        os.makedirs('./data/plots')
        
    # Save plot as image
    confusion_matrix_plot.savefig('./data/plots/' + tag + '.png', dpi=300, bbox_inches='tight')

    return 

def calc_metrics(TP, FP, FN):
    '''
    Calculates accuracy, precision, recall, and F1 score based on (TP | FP | FN) metrics.
    Returns 0 to avoid DivisionByZero exceptions
    Note that, the nature of the model does not allow true negatives (TN).
    '''
    acc = (TP) / (TP + FP + FN) if (TP + FP + FN > 0) else 0
    prec = (TP) / (TP + FP) if (TP + FP > 0) else 0
    rec = (TP) / (TP + FN) if (TP + FN > 0) else 0
    f1 = 2 * ((prec * rec) / (prec + rec)) if (prec + rec > 0) else 0
    
    return acc, prec, rec, f1
