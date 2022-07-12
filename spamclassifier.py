import numpy as np
import math


class SpamClassifier:
    def __init__(self, k):
        print("I am initializing")
        self.k = k
        self.training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
        self.decision_tree = np.empty(0)
        # #Add more 'training' data
        # extra_training_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
        # print("training_spam shape: ", self.training_spam.shape)
        # print("extra_training_spam shape: ", extra_training_spam.shape)
        # self.training_spam = np.concatenate((self.training_spam, extra_training_spam), axis=0)

        print("Shape of the spam training data set:", self.training_spam.shape)
        print(self.training_spam)

    def train(self, use_decision_tree=False):
        if use_decision_tree:
            self.decision_tree = self.train_id3_decision_tree(self.training_spam)
        else:
            self.estimate_log_class_priors(self.training_spam)
            self.estimate_log_class_conditional_likelihoods(self.training_spam)

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
        # print("count_c1: ", count_c1)
        # print("num_rows:   ", num_rows)

        log_c0_prior = math.log((num_rows - count_c1) / num_rows)
        log_c1_prior = math.log(count_c1 / num_rows)
        log_class_priors = np.array([log_c0_prior, log_c1_prior])
        # print("log_class_priors:", log_class_priors)

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

        self.log_class_conditional_likelihoods = theta

    def get_probability(self, class_num, message, log_class_priors, log_class_conditional_likelihoods):
        # print("log_class_conditional_likelihoods shape: ", log_class_conditional_likelihoods.shape)
        # print("message shape: ", message.shape)
        running_probability = 1
        i = 0
        for element in message:
            if element == 1:
                running_probability += log_class_conditional_likelihoods[class_num, i]
            i += 1

        running_probability *= log_class_priors[class_num]
        return running_probability

    def predict(self, data, use_decision_tree=False):
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
        class_predictions = np.empty(data.shape[0])

        message = 0

        if use_decision_tree:
            class_predictions = self.parse_decision_tree(data)
        else:
            for row in data:
                print("Row: ", row)
                # note from self consider making more accurate by using the inverse of the probability if a selection is NOT true for a feature and class
                class0_probability = self.get_probability(0, row, self.log_class_priors,
                                                          self.log_class_conditional_likelihoods)
                print("class0_probability: ", class0_probability)
                class1_probability = self.get_probability(1, row, self.log_class_priors,
                                                          self.log_class_conditional_likelihoods)
                print("class1_probability: ", class1_probability)
                probably_spam = class1_probability < class0_probability
                print("probably_spam: ", probably_spam)

                class_predictions[message] = probably_spam
                message += 1

        return class_predictions

    def train_id3_decision_tree(self, training_spam):
        # 'ID3 decision tree generator
        # ref https://medium.com/analytics-vidhya/entropy-calculation-information-gain-decision-tree-learning-771325d16f'
        print("test:", [i for i in range(0, training_spam.shape[1])])
        training_spam = np.vstack(([i for i in range(-1, training_spam.shape[1] - 1)],
                                   training_spam))  # add index ref header row
        return self.generate_decision_tree(training_spam)

    def generate_decision_tree(self, training_spam):
        training_spam = np.unique(training_spam, axis=0)  # remove duplicate rows
        gain_per_attribute = np.zeros((training_spam.shape[1] - 1, 2))
        max_gain = np.zeros(3)  # array used to store running highest gain
        #                                format: [index in current S, index in original, gain]

        for S in range(1, training_spam.shape[1]):
            print("checking index ", S)
            s0c0 = 0
            s0c1 = 0
            s1c0 = 0
            s1c1 = 0
            s0 = 0
            s1 = 0
            c0 = 0
            c1 = 0
            for i in range(1, training_spam.shape[0] - 1):
                if training_spam[i][S] == 0:
                    if training_spam[i][0] == 0:
                        s0c0 += 1
                    else:
                        s0c1 += 1
                else:  # S=1
                    if training_spam[i][0] == 0:
                        s1c0 += 1
                    else:
                        s1c1 += 1
            s0 = s0c0 + s0c1
            s1 = s1c0 + s1c1
            c0 = s0c0 + s1c0
            c1 = s0c1 + s1c1
            total = c0 + c1

            # return terminals if all classifications are the same
            if c0 == 0:
                return "SPAM"
            if c1 == 0:
                return "HAM"

            s_gain = self.calculate_gain(s0c0, s0c1, s1c0, s1c1, s0, s1, c0, c1, total)

            print("Gain for function column {} = {}".format(S, s_gain))

            if s_gain > max_gain[2]:
                max_gain = [S, training_spam[0, S], s_gain]
            gain_per_attribute[S - 1] = [training_spam[0, S], s_gain]
            # note later if I get to function combinations that return spam or ham check through data to see which had
            # more instances instead of accepting/rejecting outright
            pass

        s0_data = self.split_on_attribute_value(training_spam, max_gain[0], 0)
        s1_data = self.split_on_attribute_value(training_spam, max_gain[0], 1)
        # s0_rows = np.insert(np.where(training_spam[:, max_gain[0]] == 0), 0, 0)
        # s1_rows = np.insert(np.where(training_spam[:, max_gain[0]] == 1), 0, 0)
        # s0_data = np.delete(training_spam[s0_rows], max_gain[0], axis=1)
        # s1_data = training_spam[s1_rows]

        # following iterated OK but created extra array containers i.e. [49, [[52, [[18, [[3, [[14, [[19, [[21, [[....
        # next_node = [self.generate_decision_tree(s0_data),
        #              self.generate_decision_tree(s1_data)]
        # branch = [max_gain[1], next_node]

        # next_node = [self.generate_decision_tree(s0_data),
        #              self.generate_decision_tree(s1_data)]
        branch = [max_gain[1], self.generate_decision_tree(s0_data),
                  self.generate_decision_tree(s1_data)]
        return branch

    def calculate_gain(self, s0c0, s0c1, s1c0, s1c1, s0, s1, c0, c1, total):

        print("s0 = s0c0 + s0c1: ", s0)
        print("s1 = s1c0 + s1c1: ", s1)
        print("c0 = s0c0 + s1c0: ", c0)
        print("c1 = s0c1 + s1c1: ", c1)

        # lambda to protect against dividing by or applying log2 to zero,
        #       confirms neither side is zero then returns entropy instance calculation
        entropy_instance = lambda x, y: 0 if x == 0 or y == 0 else (x / y) * math.log2(x / y)

        # entropy = (c0 / total) * math.log2(c0 / total) + (c1 / total) * math.log2(c1 / total)
        entropy = -(entropy_instance(c0, total) + entropy_instance(c1, total))
        print("S_entropy: ", entropy)

        # entropy_s0 = -((s0c0 / s0) * math.log2(s0c0 / s0) + (s0c1 / s0) * math.log2(s0c1 / s0))
        entropy_s0 = -(entropy_instance(s0c0, s0)) + entropy_instance(s0c1, s0)
        print("entropy_s0: ", entropy_s0)

        # entropy_s1 = -((s1c0 / s0) * math.log2(s1c0 / s0) + (s1c1 / s0) * math.log2(s1c1 / s0))
        entropy_s1 = -(entropy_instance(s1c0, s0)) + entropy_instance(s1c1, s0)
        print("entropy_s1: ", entropy_s1)

        gain = entropy - ((s0 / total) * entropy_s0 + (s1 / total) * entropy_s1)
        print("gain: {}\n", gain)

        return gain

    def split_on_attribute_value(self, input_table, attribute_index, value):
        result_rows = np.insert(np.where(input_table[:, attribute_index] == value), 0, 0)
        result = np.delete(input_table[result_rows], attribute_index, axis=1)
        return result

    def parse_decision_tree(self, data):
        output = None
        current_branch = self.decision_tree
        current_branch_index = 0
        for row in data:
#poss move variable reset to here

            while output == None:
                current_attribute = current_branch[0]
                current_value = row[current_attribute]
                print("current_attribute {} = {}".format(current_attribute, current_value))
                if isinstance(current_value, np.int32):
                    if current_value == 0:
                        if isinstance(current_branch[1], str):
                            output = current_branch[1]
                        current_branch = current_branch[1]
                    else:  # ==1
                        if isinstance(current_branch[1][1], str):
                          output = current_branch[1][1]
                        current_branch = current_branch[1][1]
                    print("current branch: ", current_branch)
                else:
                    output = row[current_attribute]
                    print("Row is ", output)
                # add piece to place decision in output array
        # go through decision tree array for each row
        return output == "SPAM"


def create_classifier(use_decision_tree=False):
    classifier = SpamClassifier(k=1)
    classifier.train(use_decision_tree)
    print("decision tree: ", classifier.decision_tree)
    return classifier


classifier = create_classifier(True)


def run_tests():
    SKIP_TESTS = False

    if not SKIP_TESTS:
        testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
        # testing_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
        test_data = testing_spam[:, 1:]
        test_labels = testing_spam[:, 0]

        predictions = classifier.predict(test_data, True)
        accuracy = np.count_nonzero(predictions == test_labels) / test_labels.shape[0]

        were_good_index = np.where(test_labels == 0)
        were_good_results = predictions[were_good_index]
        # print(were_good_results)
        print("{} out of {} good emails were classed as spam ({:.2f}%)".format(np.sum(were_good_results),
                                                                               were_good_results.shape[0],
                                                                               np.sum(were_good_results) /
                                                                               were_good_results.shape[0]
                                                                               * 100))
        were_spam_index = np.where(test_labels == 1)
        were_spam_results = predictions[were_spam_index]
        # print(were_spam_results)
        print("{} out of {} spam emails were classed as spam ({:.2f}%)".format(np.sum(were_spam_results),
                                                                               were_spam_results.shape[0],
                                                                               np.sum(were_spam_results) /
                                                                               were_spam_results.shape[0]
                                                                               * 100))
        print(f"Accuracy on test data is: {accuracy}")


run_tests()
