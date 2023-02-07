# import random


# def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
#     print("Starting Evaluation.....")
#     output = {}
#     if phase_codename == "dev":
#         print("Evaluating for Dev Phase")
#         output["result"] = [
#             {
#                 "train_split": {
#                     "Metric1": random.randint(0, 99),
#                     "Metric2": random.randint(0, 99),
#                     "Metric3": random.randint(0, 99),
#                     "Total": random.randint(0, 99),
#                 }
#             }
#         ]
#         # To display the results in the result file
#         output["submission_result"] = output["result"][0]["train_split"]
#         print("Completed evaluation for Dev Phase")
#     return output

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    # neutral, happiness, worry, sadness, surprise, hate
    emotions = ["neutral", "happiness", "worry", "sadness", "surprise", "hate"]
    confusion_matrix = np.zeros((len(emotions), len(emotions)), dtype=int)
    for i in range(len(test_annotation_file)):
        true_emotion = test_annotation_file[i]
        pred_emotion = user_submission_file[i]
        true_index = emotions.index(true_emotion)
        pred_index = emotions.index(pred_emotion)
        confusion_matrix[true_index][pred_index] += 1
    accuracy = accuracy_score(test_annotation_file, user_submission_file)
    precision = precision_score(test_annotation_file, user_submission_file, average='macro')
    recall = recall_score(test_annotation_file, user_submission_file, average='macro')
    f1 = f1_score(test_annotation_file, user_submission_file, average='macro')
    return confusion_matrix, accuracy, precision, recall, f1

# file_path = "tweet_emotions_submit_file.csv"
# modified_file_path = "modified_text_file.csv"

# test_annotation_file = ["joy", "sadness", "joy", "fear", "surprise", "joy", "sadness", "anger"]
# user_submission_file = ["joy", "sadness", "joy", "fear", "surprise", "joy", "sadness", "fear"]
# confusion_matrix, accuracy, precision, recall, f1 = evaluate(test_annotation_file, user_submission_file)


if phase_codename == "dev":
    print("Evaluating for Dev Phase")
    output["result"] = [
        {
            "train_split": {
                "Confusion Matrix": confusion_matrix
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1_score,
            }
        }
    ]
    output["submission_result"] = output["result"][0]["train_split"]
    print("Completed evaluation for Dev Phase")
return output

