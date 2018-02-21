import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB


def cross_fold(fold=0):
    data_dir = "movie_reviews"
    classes = ['pos', 'neg']
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for curr_class in classes:
        dirname = os.path.join(data_dir, curr_class)  # data_dir\cur
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r') as f:
                content = f.read()
                if fname.startswith('cv' + str(fold)):
                    test_data.append(content)
                    test_labels.append(curr_class)
                else:
                    train_data.append(content)
                    train_labels.append(curr_class)

    print("Testing fold %d: %d" % (fold + 1, len(test_labels)))

    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    # Perform Naive Bayes classifier for multinomial models
    # http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    classifier_rbf = MultinomialNB()
    classifier_rbf.fit(train_vectors, train_labels)
    prediction_rbf = classifier_rbf.predict(test_vectors)

    # Print results in a nice table
    print("Results of Naive Bayes classifier")
    print(classification_report(test_labels, prediction_rbf))
    print("Accuracy : %.05f" % np.mean(test_labels == prediction_rbf))

    return np.mean(test_labels == prediction_rbf)


Acc = []
for f in range(0, 10):
    acc = cross_fold(f)
    Acc.append(acc)
print("Cross Validation Accuracies:  %.07f" % np.mean(Acc))
