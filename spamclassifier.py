import numpy as np
import math


class SpamClassifier:
    def __init__(self, k):
        self.log_class_conditional_likelihoods = None
        self.log_class_priors = None
        print("Classifier is initialising")
        self.k = k
        self.training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
        self.decision_tree = np.empty(0)

        print("Shape of the spam training data set:", self.training_spam.shape)
        # print(self.training_spam)

    def train(self, use_decision_tree=1):
        if use_decision_tree == 0:
            print("Training naive bayes against training data")
            self.estimate_log_class_priors(self.training_spam)
            self.estimate_log_class_conditional_likelihoods(self.training_spam)
        elif use_decision_tree == 1:
            print("Training decision tree")
            self.decision_tree = self.train_id3_decision_tree(self.training_spam)
        else:
            print("Training decision tree")
            self.decision_tree = self.train_id3_decision_tree(self.training_spam)
            print("Parsing decision tree against training data")
            decision_tree_feedback = self.parse_decision_tree(self.training_spam)
            print("Appending decision tree predictions to training data")
            enhanced_train_data = np.zeros(self.training_spam.shape[1] + 1)
            for x in zip(self.training_spam, decision_tree_feedback):
                # print("combined row: ", np.append(x[0], x[1]))
                enhanced_train_data = np.vstack((enhanced_train_data, np.append(x[0], x[1])))
            enhanced_train_data = np.delete(enhanced_train_data, (0), axis=0)
            print("Enhanced training data shape: ", enhanced_train_data.shape)
            print("Training naive bayes against enhanced data")
            self.estimate_log_class_priors(enhanced_train_data)
            self.estimate_log_class_conditional_likelihoods(enhanced_train_data)
            print("Naive bayes trained against enhanced data")

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

        num_rows = data.shape[0]
        count_c1 = np.sum(data[:, 0])

        log_c0_prior = math.log((num_rows - count_c1) / num_rows)
        print("log_c0_prior: ", log_c0_prior)
        log_c1_prior = math.log(count_c1 / num_rows)
        print("log_c1_prior: ", log_c1_prior)
        log_class_priors = np.array([log_c0_prior, log_c1_prior])

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
        :param alpha: indicator used to determine whether laplace smoothing is used as per
                        project requirements

        :return theta:
            a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
            logarithm of the probability of feature i appearing in a sample belonging
            to class j.
        """

        c0_element_eccl = self.estimate_single_log_class_conditional_likelihood(data, 0, alpha)
        c1_element_eccl = self.estimate_single_log_class_conditional_likelihood(data, 1, alpha)

        theta = np.array([c0_element_eccl, c1_element_eccl])
        # print("theta: ", theta)
        self.log_class_conditional_likelihoods = theta

    def estimate_single_log_class_conditional_likelihood(self, data, c_index, alpha):
        # put together to create a 1D array of the sum of each element where C=1 (tested and correct in excel)
        c_rows = data[np.where(data[:, 0] == c_index)[0], 1:]
        c_numrows = c_rows.shape[0]
        c_element_sums = np.array(np.sum(c_rows, axis=0))
        c_total_words = np.sum(c_element_sums)
        if alpha > 0:  # add 1 to every zero element and its denominator for laplace smoothing
            for i in range(c_element_sums.shape[0]):
                if c_element_sums[i] == 0:
                    c_element_sums[i] = 1
            # c1_element_sums = np.add(1, c1_element_sums)
            c_denominator = c_total_words + c_element_sums.shape[0]
        else:
            c1_denominator = c_total_words + c_element_sums.shape[0]

        c_element_eccl = np.log(np.divide(c_element_sums, c_denominator))
        return c_element_eccl

    def run_naive_bayes(self, data):
        class_predictions = np.empty(data.shape[0])
        message = 0

        for row in data:
            class0_probability = np.sum(self.log_class_conditional_likelihoods[0] * row) + self.log_class_priors[0]
            class1_probability = np.sum(self.log_class_conditional_likelihoods[1] * row) + self.log_class_priors[1]
            class_predictions[message] = class1_probability > class0_probability
            message += 1
        return class_predictions

    def predict(self, data, use_decision_tree=1):
        """
        Given a new data set with binary features, predict the corresponding
        response for each instance (row) of the new_data set.

        :param use_decision_tree: integer value to decide what form of test to run (must match
            the form chosen in train method)
            0 = Naive Bayes
            1 = Decision Tree (default)
            2 = Decision Tree and Naive Bayes
        :param data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].

        :return class_predictions: a numpy array containing the class predictions for each row
            of new_data.
        """

        self.train(use_decision_tree)
        class_predictions = np.empty(data.shape[0])
        message = 0

        if use_decision_tree == 0:
            print("Running naive bayes against test data")
            class_predictions = self.run_naive_bayes(data)
        elif use_decision_tree == 1:
            print("Parsing decision tree against test data")
            class_predictions = self.parse_decision_tree(data)
        else:
            print("Parsing decision tree against test data")
            class_predictions = self.parse_decision_tree(data)
            print("Adding decision tree predictions to test data")
            enhanced_test_data = np.zeros(data.shape[1] + 1)
            for x in zip(data, class_predictions):
                enhanced_test_data = np.vstack((enhanced_test_data, np.append(x[0], x[1])))
            enhanced_test_data = np.delete(enhanced_test_data, (0), axis=0)
            print("Running naive bayes against enhanced test data")
            class_predictions = self.run_naive_bayes(enhanced_test_data)

        return class_predictions

    def train_id3_decision_tree(self, training_spam):
        # 'ID3 decision tree generator
        # ref https://medium.com/analytics-vidhya/entropy-calculation-information-gain-decision-tree-learning-771325d16f'
        training_spam = np.vstack(([i for i in range(-1, training_spam.shape[1] - 1)],
                                   training_spam))  # add index ref header row
        return self.generate_decision_tree(training_spam)

    def generate_decision_tree(self, training_spam):
        training_spam = np.unique(training_spam, axis=0)  # remove duplicate rows
        gain_per_attribute = np.zeros((training_spam.shape[1] - 1, 2))
        max_gain = np.zeros(3)  # array used to store running highest gain
        #                                format: [index in current S, index in original, gain]
        # print("calculating max gain for table:\n", training_spam)
        for S in range(1, training_spam.shape[1]):
            # print("checking index ", S)
            s0c0 = s0c1 = s1c0 = s1c1 = s0 = s1 = c0 = c1 = 0
            for i in range(1, training_spam.shape[0] - 1):
                if training_spam[i][S] == 0:
                    if training_spam[i][0] == 0:
                        s0c0 += 1
                    else:  # c=1
                        s0c1 += 1
                else:  # S=1
                    if training_spam[i][0] == 0:
                        s1c0 += 1
                    else:  # c=1
                        s1c1 += 1
            s0 = s0c0 + s0c1
            s1 = s1c0 + s1c1
            c0 = s0c0 + s1c0
            c1 = s0c1 + s1c1
            total = c0 + c1

            # Decide ham or spam based on majority rules if only 1 column at this point
            if training_spam.shape[1] == 2:
                return c1 > c0  #

            # return terminals if all classifications are the same
            if c0 == 0:
                return True  # "SPAM"
            if c1 == 0:
                return False  # "HAM"

            s_gain = self.calculate_gain(s0c0, s0c1, s1c0, s1c1, s0, s1, c0, c1, total)
            # print("Gain for function column {} = {}".format(S, s_gain))

            if s_gain > max_gain[2]:
                max_gain = [S, training_spam[0, S], s_gain]
            gain_per_attribute[S - 1] = [training_spam[0, S], s_gain]
        # print("about to split data - attribute with max gain = {} in index {}".format(max_gain[1], max_gain[0]))

        # if there is no entropy on any attribute (for example all attributes have zero
        #       positive values), no gain will have been selected - just select the first attribute.
        if max_gain[0] == 0.0:
            max_gain = [1, training_spam[0, 1], 0]

        s0_data = self.split_on_attribute_value(training_spam, max_gain[0], 0)
        s1_data = self.split_on_attribute_value(training_spam, max_gain[0], 1)

        branch = [max_gain[1], self.generate_decision_tree(s0_data),
                  self.generate_decision_tree(s1_data)]
        return branch

    def calculate_gain(self, s0c0, s0c1, s1c0, s1c1, s0, s1, c0, c1, total):

        # print("s0({}) = s0c0({}) + s0c1({})".format(s0, s0c0, s0c1))
        # print("s1({}) = s1c0({}) + s1c1({})".format(s1, s1c0, s1c1))
        # print("c0({}) = s0c0({}) + s1c0({})".format(c0, s0c0, s1c0))
        # print("c1({}) = s0c1({}) + s1c1({})".format(c1, s0c1, s1c1))

        # lambda to protect against dividing by or applying log2 to zero,
        #       confirms neither side is zero then returns entropy instance calculation
        entropy_instance = lambda x, y: 0 if x == 0 or y == 0 else (x / y) * math.log2(x / y)

        # entropy = (c0 / total) * math.log2(c0 / total) + (c1 / total) * math.log2(c1 / total)
        entropy = -(entropy_instance(c0, total) + entropy_instance(c1, total))

        # entropy_s0 = -((s0c0 / s0) * math.log2(s0c0 / s0) + (s0c1 / s0) * math.log2(s0c1 / s0))
        entropy_s0 = -(entropy_instance(s0c0, s0) + entropy_instance(s0c1, s0))

        # entropy_s1 = -((s1c0 / s0) * math.log2(s1c0 / s0) + (s1c1 / s0) * math.log2(s1c1 / s0))
        entropy_s1 = -(entropy_instance(s1c0, s1) + entropy_instance(s1c1, s1))

        gain = entropy - (((s0 / total) * entropy_s0) + ((s1 / total) * entropy_s1))

        return gain

    def split_on_attribute_value(self, input_table, attribute_index, value):
        # If there are only 3 columns left there are not enough left to split - simply delete the other column
        if input_table.shape[1] == 3:
            result = np.delete(input_table, value + 1, axis=1)
            return result
        # print("input table to split on value {} in attribute{}:\n{}".format(value, attribute_index, input_table))
        result_rows = np.insert(np.where(input_table[:, attribute_index] == value), 0, 0)
        result = np.delete(input_table[result_rows], attribute_index, axis=1)
        return result

    def parse_decision_tree(self, data):
        output = []
        jj = -1
        for row in data:
            jj += 1
            row_output = None
            current_branch = self.decision_tree

            while row_output == None:
                current_attribute = current_branch[0]
                current_value = row[current_attribute]
                if isinstance(current_value, np.int32):
                    if current_value == 0:
                        if isinstance(current_branch[1], bool):
                            row_output = current_branch[1]
                        current_branch = current_branch[1]
                    else:  # ==1
                        if isinstance(current_branch[2], bool):
                            row_output = current_branch[2]
                        else:
                            current_branch = current_branch[2]
                else:
                    row_output = row[current_attribute]
                if row_output is not None:
                    output.append(int(row_output is True))
                    # print("prediction is SPAM: ", row_output)
        return np.asarray(output)


def create_classifier():
    classifier = SpamClassifier(k=1)
    return classifier


classifier = create_classifier()


def run_tests():
    SKIP_TESTS = False

    if not SKIP_TESTS:
        testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
        # testing_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
        test_data = testing_spam[:, 1:]
        test_labels = testing_spam[:, 0]

        # classifier = create_classifier(True)
        predictions = classifier.predict(test_data, 0)
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

        # for i in range(predictions.size):
        #     print("Index:{}  Prediction:{}  actual:{}".format(i, predictions[i], test_labels[i]))


run_tests()
