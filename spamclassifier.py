import numpy as np
import math

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
        self.training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
        print("Shape of the spam training data set:", self.training_spam.shape)
        print(self.training_spam)
        # self.log_class_priors = np.array(np.empty())
        # self.log_class_conditional_likelihoods = np.array(np.empty())

    def train(self):
        self.estimate_log_class_priors(self.training_spam)
        self.estimate_log_class_conditional_likelihoods(self.training_spam)
        pass

    def estimate_log_class_priors(self, data):
        """
        Given a data set with binary response variable (0s and 1s) in the
        left-most column, calculate the logarithm of the empirical class priors,
        that is, the logarithm of the proportions of 0s and 1s:
            log(p(C=0)) and log(p(C=1))

        :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
                     the first column contains the binary response (coded as 0s and 1s).

        :return log_class_priors: a numpy array of length two
        """

        num_rows = data.shape[0] + 1  # +1 for laplace smoothing
        count_c1 = np.sum(data[:, 0]) + 1

        print("count_c1: ", count_c1)
        print("num_rows:   ", num_rows)

        log_c0_prior = math.log((num_rows - count_c1) / num_rows)
        log_c1_prior = math.log(count_c1 / num_rows)
        log_class_priors = np.array([log_c0_prior, log_c1_prior])
        print("log_class_priors:", log_class_priors)

        # return log_class_priors
        self.log_class_priors = log_class_priors

    def estimate_log_class_conditional_likelihoods(self, data, alpha=1.0):
        """
        Given a data set with binary response variable (0s and 1s) in the
        left-most column and binary features (words), calculate the empirical
        class-conditional likelihoods, that is,
        log(P(w_i | c)) for all features w_i and both classes (c in {0, 1}).

        Assume a multinomial feature distribution and use Laplace smoothing
        if alpha > 0.

        :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]

        :return theta:
            a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
            logarithm of the probability of feature i appearing in a sample belonging
            to class j.
        """
        import math

        # print(data.shape)
        # num_classes = data.shape[0]
        # num_features = data.shape[1]

        # following line returns an array of all rows with a 1 in the first column minus the first column itself, but oddly puts this in a new array of shape (1, numrows_notzero, num_features)
        # data[np.nonzero(data[:,0]),1:]

        # print the above for testing
        # with np.printoptions(threshold=np.inf):
        #    print("test",data[np.nonzero(data[:,0]),:])

        # c1_rows = data[np.nonzero(data[:,0]),1:]
        # c1_rows = np.nonzero(data[:,0])
        # print(c1_rows[0,:])
        # print("c1_rows shape: ", c1_rows[0,:].shape)
        # with np.printoptions(threshold=np.inf):
        #    print(c1_rows, "\n")

        # sum the non zero columns using numpy axis 0 (ref: https://www.pythonpool.com/numpy-axis/)
        # c1_element_sums = np.array(np.sum(c1_rows[0,:], axis = 0))
        # print(c1_element_sums)
        # print("c1_element_sums shape: ", c1_element_sums.shape)
        # print("c1_element_sums: ", c1_element_sums)

        ##############################################
        # put together to create a 1D array of the sum of each element where C=1 (tested and correct in excel)
        c1_rows = data[np.nonzero(data[:, 0]), 1:]
        c1_numrows = c1_rows[0, :, 0].size
        # print("c1_numrows: ", c1_numrows)
        c1_element_sums = np.array(np.sum(c1_rows[0, :], axis=0))
        if alpha > 0:
            c1_element_sums = np.add(1, c1_element_sums)  # add 1 to every element fpr laplace smoothing
        # print("c1_element_sums,",c1_element_sums)
        c1_element_eccl = np.log(np.divide(c1_element_sums, c1_numrows))
        # print("c1_element_eccl: ", c1_element_eccl, c1_element_eccl.shape)
        # print("Test numpy log10 0.147: ", np.log10(0.147))
        # print("Test math log 0.147: ", math.log(0.147))
        ##################################################

        # THINK I HAVE THIS WRONG - I SHOULD BE COUNTING THE PROBABILITY OF EACH ELEMENT BEING 1 FOR A CLASS, BUT THINK I AM COUNTING THE ELEMENTS WHICH ARE ZERO FOR EACH CLASS
        # now get rows where C=0 using idea from https://stackoverflow.com/questions/4588628/find-indices-of-elements-equal-to-zero-in-a-numpy-array
        #################################################
        c0_rows = data[np.where(data[:, 0] == 0)[0], 1:]
        # with np.printoptions(threshold=np.inf):
        #    print("c0_rows", c0_rows.shape, "\n", c0_rows)
        c0_numrows = c0_rows[:, 0].size
        # print("c0_numrows: ", c0_numrows)
        c0_element_sums = np.array(np.sum(c0_rows, axis=0))
        if alpha > 0:
            c0_element_sums = np.add(1, c0_element_sums)  # add 1 to every element fpr laplace smoothing
        # print("c0_element_sums,",c0_element_sums)
        c0_element_eccl = np.log(np.divide(c0_element_sums, c0_numrows))
        # print("c0_element_eccl: ", c0_element_eccl, c0_element_eccl.shape)

        theta = np.array([c0_element_eccl, c1_element_eccl])
        # print("theta: ", theta, theta.shape)

        # return theta
        self.log_class_conditional_likelihoods = theta

    def get_probability(self, class_num, message, log_class_priors, log_class_conditional_likelihoods):
        print("log_class_conditional_likelihoods shape: ", log_class_conditional_likelihoods.shape)
        print("message shape: ", message.shape)
        running_probability = 1
        i = 0
        for element in message:
            if element == 1:
                running_probability *= log_class_conditional_likelihoods[class_num, i]
            else:
                # print("original probability: ", log_class_conditional_likelihoods[class_num,i])
                # print("inverse log:", math.e ** log_class_conditional_likelihoods[class_num,i])
                # print("opposite probability:", 1-(math.e ** log_class_conditional_likelihoods[class_num,i]))
                # print("opposite log probability:", math.log(1-(math.e ** log_class_conditional_likelihoods[class_num,i])))
                running_probability *= math.log(1 - (math.e ** log_class_conditional_likelihoods[class_num, i]))
            i += 1

        denominator = log_class_priors[class_num]
        # print("denominator:", denominator)
        # perform calculation
        probability = (running_probability / denominator)
        return probability

    # def predict(self, data):
    #     return np.zeros(data.shape[0])
    def predict(self, data):
        """
        Given a new data set with binary features, predict the corresponding
        response for each instance (row) of the new_data set.

        :param data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].
        :param log_class_priors: a numpy array of length 2.
        :param log_class_conditional_likelihoods: a numpy array of shape = [2, n_features].
            theta[j, i] corresponds to the logarithm of the probability of feature i appearing
            in a sample belonging to class j.
        :return class_predictions: a numpy array containing the class predictions for each row
            of new_data.
        """
        ### YOUR CODE HERE...
        class_predictions = np.empty(data.shape[0])

        message = 0
        for row in data:
            print("Row: ", row)
            # calculate numerator
            # note from self consider making more accurate by using the inverse of the probability if a selection is NOT true for a feature and class
            class0_probability = self.get_probability(0, row, self.log_class_priors,
                                                      self.log_class_conditional_likelihoods)
            print("class0_probability: ", class0_probability)
            class1_probability = self.get_probability(1, row, self.log_class_priors,
                                                      self.log_class_conditional_likelihoods)
            print("class1_probability: ", class1_probability)
            probably_spam = class1_probability > class0_probability
            print("probably_spam: ", probably_spam)

            #        running_probability_1 = 1
            #        i=0
            #        for element in row:
            #            if element == 1:
            #                running_probability_1 *= log_class_conditional_likelihoods[1,i]
            #            i+=1

            # get log class prior (denominator)
            #        denominator = log_class_priors[1]
            #        print("denominator:", denominator)
            # perform calculation
            #        probability = (running_probability / denominator)
            # append to class predicions
            #       print("probability: ", probability)

            # class_predictions = np.append(class_predictions, probability)

            class_predictions[message] = probably_spam
            message += 1

            # COMMENT OUT LATER
            # if message > 1:
            #    return class_predictions

        return class_predictions


def create_classifier():
    classifier = SpamClassifier(k=1)
    classifier.train()
    return classifier


classifier = create_classifier()


def run_tests():
    SKIP_TESTS = False

    if not SKIP_TESTS:
        # testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
        testing_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
        test_data = testing_spam[:, 1:]
        test_labels = testing_spam[:, 0]

        predictions = classifier.predict(test_data)
        accuracy = np.count_nonzero(predictions == test_labels) / test_labels.shape[0]
        print(f"Accuracy on test data is: {accuracy}")


run_tests()
