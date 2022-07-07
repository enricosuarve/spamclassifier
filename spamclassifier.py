import numpy as np

# training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
# print("Shape of the spam training data set:", training_spam.shape)
# print(training_spam)


testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
print("Shape of the spam testing data set:", testing_spam.shape)
print(testing_spam)


# This skeleton code simply classifies every input as ham
#
# Here you can see there is a parameter k that is unused, the
# point is to show you how you could set up your own. You might
# also pass in extra data via a train method (also does nothing
#  here). Modify this code as much as you like so long as the
# accuracy test in the cell below runs.

class SpamClassifier:
    def __init__(self, k):
        print("I am initializing")
        self.k = k
        training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
        print("Shape of the spam training data set:", training_spam.shape)
        print(training_spam)

        np.array()

    def train(self):
        pass

    def predict(self, data):
        return np.zeros(data.shape[0])

    # def load_training_data(self):




def create_classifier():
    classifier = SpamClassifier(k=1)
    classifier.train()
    return classifier


classifier = create_classifier()


def run_tests():
    SKIP_TESTS = False

    if not SKIP_TESTS:
        testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
        test_data = testing_spam[:, 1:]
        test_labels = testing_spam[:, 0]

        predictions = classifier.predict(test_data)
        accuracy = np.count_nonzero(predictions == test_labels) / test_labels.shape[0]
        print(f"Accuracy on test data is: {accuracy}")


run_tests()
