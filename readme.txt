There are 33 different modes to run the classifier in

0 - Naïve Bayes
1 - Decision Tree (the default)
2 - Decision Tree prediction fed back into data then Naïve Bayes applied to the result

The mode is decided via an additional attribute 'use_decision_tree' in the predict method

For example to run the test in Naïve Bayes mode the following syntax should be used:

    predictions = classifier.predict(test_data, 0)

If no value is entered for use_decision_tree, the classifier will default to decision tree as this has given the best results during testing.