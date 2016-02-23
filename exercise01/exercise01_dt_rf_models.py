# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier as rf


# Function to create the ROC/AUC curve, and display the auc
def plot_roc(fpr, tpr, title):
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC ' + title)
    plt.legend(loc="lower right")
    plt.show()


##################################
#       DATA PREPARATION         #
##################################

# Moving over to python, I will try some machine learning/sklearn modeling.
# First, read in the flattened database
records_train = pd.read_csv("records_train.csv", sep = ",")
records_valid = pd.read_csv("records_valid.csv", sep = ",")
records_test = pd.read_csv( "records_test.csv", sep = ",")

# Confirm the import worked correctly
records_train.describe()
records_train.head()
records_valid.describe()
records_valid.head()
records_test.describe()
records_test.head()

# Separate out target variable
records_train_y = records_train.over_50k
records_valid_y = records_valid.over_50k
records_test_y = records_test.over_50k

# Based on google searching, in order to use the decision tree
# classifier with categorical variables, I will need to separate
# the variable types, create a DictVectorizer object, and then
# bring the variables back together as a numpy stack. This
# transformation process was found here:
# http://fastml.com/converting-categorical-data-into-numbers-
# with-pandas-and-scikit-learn/

# First, separate the numeric columns
numeric_cols = ['age', 'education_num', 'capital_gain', 'capital_loss']
records_train_x_num = records_train[numeric_cols].as_matrix()
records_valid_x_num = records_valid[numeric_cols].as_matrix()
records_test_x_num = records_test[numeric_cols].as_matrix()

# Next, the categorical columns
categorical_cols = ['marital_status', 'occupation', 'race', 'relationship', 'sex']
cat_train = records_train[categorical_cols]
cat_valid = records_valid[categorical_cols]
cat_test = records_test[categorical_cols]

cat_train.fillna('NA', inplace=True)
cat_valid.fillna('NA', inplace=True)
cat_test.fillna('NA', inplace=True)

# Using transposition of the columns to rows, I can use the to_dict.values
# to get the "row" dictionary values.
cat_dict_train = cat_train.T.to_dict().values()
cat_dict_valid = cat_valid.T.to_dict().values()
cat_dict_test = cat_test.T.to_dict().values()

# Run the vectorizer, and create "horizontal stacks"
vectorizer = DictVectorizer(sparse = False)
records_train_x_vec = vectorizer.fit_transform(cat_dict_train)
records_valid_x_vec = vectorizer.transform(cat_dict_valid)
records_test_x_vec = vectorizer.transform(cat_dict_test)

records_train_x_final = np.hstack(( records_train_x_num, records_train_x_vec ))
records_valid_x_final = np.hstack(( records_valid_x_num, records_valid_x_vec ))
records_test_x_final = np.hstack(( records_test_x_num, records_test_x_vec ))


##################################
#        DECISION TREES          #
##################################

# Create a decision tree with a max depth of 3 based on highest entropy.
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=3, random_state=0)

tree.fit(records_train_x_final, records_train_y)

valid_tree1_pred = tree.predict(records_valid_x_final)

fpr, tpr, thresholds = roc_curve(records_valid_y, valid_tree1_pred)
plot_roc(fpr, tpr, "Decision Tree")

# A value of 0.7202 isn't bad for a max depth of three.

export_graphviz(tree, out_file='tree1.dot')


##### GRID SEARCH #####
# Next, I will utilize the grid search to create multiple decision
# trees and determine the best predictor.

tree2 = DecisionTreeClassifier(criterion = 'entropy')
parameters = {'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
gridsearch = GridSearchCV(tree2, parameters)
gridsearch.fit(records_train_x_final, records_train_y)

# see what is the best max_depth
print("The depth for the most accurate model was " + str(gridsearch.best_estimator_.max_depth))

# create predictions
records_valid_y_pred_clf = gridsearch.predict(records_valid_x_final)

fpr, tpr, thresholds = roc_curve(records_valid_y, records_valid_y_pred_clf)
plot_roc(fpr, tpr, "DT Grid Search")

# A value of 0.7447 is better, but lower than the all other models.

# Using the value of 9 for the test set
tree_test = DecisionTreeClassifier(criterion = 'entropy', max_depth=9, random_state=0)

tree_test.fit(records_train_x_final, records_train_y)

records_test_y_pred = tree_test.predict(records_test_x_final)

fpr, tpr, thresholds = roc_curve(records_test_y, records_test_y_pred)
plot_roc(fpr, tpr, "DT Grid Search (Test Set)")

# Increased again to 0.7549, showing the grid search is helpful, but not
# optimal.

export_graphviz(tree_test, out_file='tree_test.dot')


##################################
#         RANDOM FOREST          #
##################################

# Lastly, I will look at the random forest model, and see how well it predicts
# the validation set.
rforest = rf(n_estimators=100, max_features='auto', verbose=1, n_jobs=1)
rforest.fit(records_train_x_final, records_train_y)


rf_probabilities = rforest.predict_proba(records_valid_x_final)

roc_auc = roc_auc_score(records_valid_y, rf_probabilities[:,1] )
fpr, tpr, thresholds = roc_curve(records_valid_y, rf_probabilities[:,1])

plot_roc(fpr, tpr, "Random Forest")ppyp
