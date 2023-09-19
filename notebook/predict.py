import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from io import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score


def make_logistic_predictions(pdata,cvalue = 1.0):
    data = get_filtered_data(pdata)
    X=  data.drop(columns=['foreclosured'])
    y = data['foreclosured']
  
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, stratify=y)
    model = LogisticRegression(C= cvalue,random_state=1, class_weight="balanced",max_iter = 1000)
    predictions = cross_val_score(model,Xtrain,ytrain,cv=3)
    print("LogicsticRegression: Cross-Validation Scores:", predictions)
    print("Mean CV Score:", predictions.mean())
    print("LogicsticRegression: Standard Deviation of CV Scores:", predictions.std())
    model.fit(Xtrain, ytrain)
    test_accuracy = model.score(Xtest, ytest)
    print("LogicsticRegression : Test Set Accuracy:", test_accuracy)
    ypred = model.predict(Xtest)
    print('Confusion matrix')
    print(confusion_matrix(ytest,ypred))
    print('************* LogicsticRegression Classification report ***************************')
    print(classification_report(ytest,ypred))
    print('***********************************************************************************')
    print(f'LogicsticRegression Accuracy score " {accuracy_score(ytest,ypred)}' )
    
    return
    
    
def Cvalue_section(pdata):
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]

    data = get_filtered_data(pdata)
    X=  data.drop(columns=['foreclosured'])
    y = data['foreclosured']
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, stratify=y)
    
    # Store mean cross-validation scores for each C value
    cv_scores = []
  
    for C in C_values:
        model = LogisticRegression(C=C, random_state=1, class_weight="balanced", max_iter=1000)
        scores = cross_val_score(model, Xtrain, ytrain, cv=3)  # You can adjust the number of folds
        mean_score = scores.mean()
        cv_scores.append(mean_score)
      
    optimal_C = C_values[cv_scores.index(max(cv_scores))]
    return optimal_C

def make_LogicsticRegression_scaled(pdata):
    print('************* LogicsticRegression Classification report for scaled data***************************')
    data = get_filtered_data(pdata)
        
    X=  data.drop(columns=['foreclosured'])
    y = data['foreclosured'] 
      
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, stratify=y)
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    Xtrain_scaled = scaler.transform(Xtrain)
    Xtests_scaled = scaler.transform(Xtest)
    model = LogisticRegression(random_state=1, class_weight="balanced",max_iter = 1000)
    predictions = cross_val_score(model,Xtrain_scaled,ytrain,cv=3)
    
    print("LogicsticRegression: Cross-Validation Scores -- scaled data:", predictions)
    print("Mean CV Score:", predictions.mean())
    print("LogicsticRegression: Standard Deviation of CV Scores - scaled data:", predictions.std())
    model.fit(Xtrain_scaled, ytrain)
    test_accuracy = model.score(Xtests_scaled, ytest)
    print("Test Set Accuracy:", test_accuracy)
    ypred = model.predict(Xtests_scaled)
    print('Confusion matrix')
    print(confusion_matrix(ytest,ypred))
    
    print(classification_report(ytest,ypred))
    print('*************************************************************************************************')
    print(f'LogicsticRegression Accuracy score -- scaled data " {accuracy_score(ytest,ypred)}' )
    return

def feature_selection(pdata):
        k = 2
        subset = pdata[[
        "interest_rate",
        "borrowers_count",
        "originalLoanAmt",
        "loan_term",
        "dti",
        "borrower_credit_score", 
        "first_time_homebuyer",
        "cltv",              
        "foreclosured",
        "channel",        
        "loan_purpose",
        "property_type",
        "property_state_n"]]
        
        #subset["high_balance_loan_ind"] = subset["high_balance_loan_ind"].map({'Y':1,'N':0})
        #subset["relocation_indicator"] = subset["relocation_indicator"].map({'Y':1,'N':0})

        subset = subset.dropna(axis=0, how='any')
        X= subset.drop(columns = ['foreclosured'])
        y = subset['foreclosured']  

        feature_selector = SelectKBest(score_func=chi2, k= k)
        X_new = feature_selector.fit_transform(X, y)     
        selected_feature_indices = feature_selector.get_support(indices=True)

        column_names = X.columns.tolist() 
        selected_feature_names = [column_names[i] for i in selected_feature_indices]

        # Print the indices of the selected features
        print("Selected feature indices:", selected_feature_indices)
        print("Selected feature indices:", selected_feature_names)
        return
        
def get_filtered_data(data):
        filtered_data = data[[  
        "loanId",
        "originalLoanAmt",
        "interest_rate",
        "borrowers_count",
        "dti",
       # "loan_term",
        "cltv",
        "ltv",
       # "seller",
       # "channel",
        "borrower_credit_score", 
        "high_balance_loan_ind",
        "first_time_homebuyer",
        "property_type",
        #"property_state",
        "foreclosured"
        ]]
       # filtered_data["first_time_homebuyer"]= filtered_data["first_time_homebuyer"].map({'Y':1 ,'N':0})
        
        filtered_data = filtered_data.dropna(axis=0, how='any')
        print("****filtered_data*****")

        return filtered_data
    
def make_decision_tree_predictions(pdata):
    data = get_filtered_data(pdata)
        
    X=  dataset.drop(columns=['foreclosured'])
    y = dataset['foreclosured'] 
        
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.2, random_state = 42)
    clf = DecisionTreeClassifier(class_weight ='balanced', criterion ='entropy')
    clf.fit(Xtrain, ytrain)
    
    # Define the hyperparameter grid to search
    param_grid = {
      'max_depth': [None, 10, 20, 30, 40, 50],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4]     
     }

     # Create a RandomizedSearchCV object
    random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_grid,
    n_iter=100,  # Number of random combinations to try
    cv=5,         # Number of cross-validation folds
    scoring='accuracy',  # Use an appropriate scoring metric
    n_jobs=-1,    # Use all available CPU cores for parallelism
    random_state=42  # Set a random seed for reproducibility
    )

    random_search.fit(X, y)  # Replace X and y with your data

    # Get the best hyperparameters
    best_params = random_search.best_params_
    print("Best Hyperparameters -- for DecisionTree Randomized Search:", best_params)
    feats = {} # a dict to hold feature_name: feature_importance
    for feature, importance in zip(X.columns, classifier.feature_importances_):
        feats[feature] = importance #add the name/value pair 
    print("**********************Features with importance *******************************")
    print(feats)
    best_fit = random_search.best_estimator_
    
    predictions = best_fit.predict(Xtest)
    prediction_prob = best_fit.predict_proba(Xtest)
    print(prediction_prob.shape)
  
    print('Decision tree: Confusion matrix')
    print(confusion_matrix(ytest,predictions))
    print('********Decision tree, with randomized Search Classification report ***************************')
    print(classification_report(ytest,predictions))
    print('***************************************************************************************************')
    print(f'Decision tree Classifer tree: Accuracy score " {accuracy_score(ytest,predictions)}' )
    
    return 

def best_random_forest_parameters(pdata):    
    filtered_data =    get_filtered_data(pdata)
    X=  filtered_data.drop(columns=['foreclosured'])
    y = filtered_data['foreclosured'] 
        # filtered_data["first_time_homebuyer"]= filtered_data["first_time_homebuyer"].map({'Y':1 ,'N':0})
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, stratify=y)
    filtered_data = filtered_data.dropna(axis=0, how='any')
        
    param_grid = {
         'n_estimators': [10, 20, 30],
         'max_depth': [None, 10, 20],
         'min_samples_split': [2, 5, 10],
         'min_samples_leaf': [1, 2, 4],
         'criterion': ['gini', 'entropy'],
    }
    classifier = RandomForestClassifier(n_estimators =20, random_state = 1, n_jobs = -1,class_weight='balanced')
    model_res = classifier.fit(Xtrain,ytrain)
        #grid = make_rand_forest_grid()
    kfold = model_selection.KFold(n_splits=5)
    CV_rfc = RandomizedSearchCV(estimator = classifier,
                                param_distributions = param_grid,
                                n_iter = 10,
                                cv = kfold,
                                random_state=1,
                                n_jobs = -1)
    CV_rfc.fit(Xtrain, ytrain)
    params = CV_rfc.best_params_
    print("Best parameters for random forest classification are ",  params)
    feats = {} # a dict to hold feature_name: feature_importance
    for feature, importance in zip(X.columns, classifier.feature_importances_):
        feats[feature] = importance #add the name/value pair 
    print("**********************Features with importance *******************************")
    print(feats)
   
    return feats, params
  
    
def make_random_forest_predictions (pdata,parameters,columnnames):
    #print("calling the function *****")
    
    filtered_data =  get_filtered_data(pdata)
    print(f'foreclosured loans in filtered_data : {filtered_data[filtered_data["foreclosured"] == 1]["foreclosured"].count()}')
    print(f'Good loans in filtered_data : {filtered_data[filtered_data["foreclosured"] == 0]["foreclosured"].count()}')
    X=  filtered_data[columnnames]
    print("selected columns for forclosure prediction")
    print(columnnames)
    y = filtered_data['foreclosured'] 
        # filtered_data["first_time_homebuyer"]= filtered_data["first_time_homebuyer"].map({'Y':1 ,'N':0})
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, stratify=y) 
        
    classifier = RandomForestClassifier(n_estimators = parameters['n_estimators'][0], min_samples_split = parameters['min_samples_split'][0],min_samples_leaf = parameters['min_samples_leaf'][0], max_depth = parameters['max_depth'][0], criterion =parameters['criterion'][0], random_state = 1,n_jobs = -1,class_weight= 'balanced' )
    classifier.fit(Xtrain,ytrain)
                                                                          
    predictions_train = classifier.predict(Xtrain)      
    print('***************************************************************************************************')
    print('RandomForest Classifier : Confusion matrix -- training data')
    print(confusion_matrix(ytrain,predictions_train))
    print('***RandomForest Classifer, with randomized Search Classification report training data ****')
    print(classification_report(ytrain,predictions_train))
    print(f'RandomForest Classifier tree: Accuracy score  training data : " {accuracy_score(ytrain,predictions_train)}' )
    print('***************************************************************************************************')
    print('\n\n')
    print('-----------------------------------------------------------------------------------------------------')
    thresholds = [0.5,0.55,0.6,0.65,0.7,0.8]
    acurracy_scores  = []
    for threshold in thresholds:
        prediction_prob_train = classifier.predict_proba(Xtrain)
        binary_predictions_train = (prediction_prob_train[:,1] >= threshold).astype(int)
        acurracy_scores.append(accuracy_score(ytrain,binary_predictions_train))
            
    optimal_threshold = thresholds[np.argmax(acurracy_scores)]
    print(f'Optimal thresh hold value -- {optimal_threshold}')
    print('-----------------------------------------------------------------------------------------------------')
    print('\n\n')
    predictions = classifier.predict(Xtest)
    prediction_prob_test = classifier.predict_proba(Xtest)
    
    #binary_predictions_test = (prediction_prob_test[:,1] >= optimal_threshold).astype(int)
    #print('************************************************************************************************')
    #print('RandomForest Classifer - Test Data: Confusion matrix')
    #print(confusion_matrix(ytest,binary_predictions_test))
    #print('****** RandomForest Classifer, with randomized Search Classification report ***Test Data ****')
    #print(classification_report(ytest,binary_predictions_test))    
    #print('\n')
    #print(f'RandomForest Classifer tree: Accuracy score - Test Data : {accuracy_score(ytest,predictions)}' )
    #print('***********************************************************************************************')
    
    print('************************************************************************************************')
    print('RandomForest Classifer - Test Data: Confusion matrix')
    print(confusion_matrix(ytest,predictions))
    print('****** RandomForest Classifer, with randomized Search Classification report ***Test Data ****')
    print(classification_report(ytest,predictions))    
    print('\n')
    print(f'RandomForest Classifer tree: Accuracy score - Test Data : {accuracy_score(ytest,predictions)}' )
    print('***********************************************************************************************')    
    
        
    fpr,tpr,thresholds = roc_curve(ytest,predictions)
    roc_auc = roc_auc_score(ytest, predictions)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (sensitivity)')
    plt.title('Receiver Operating Characteristic (1- specificity)')
    plt.legend(loc='lower right')
    
    for i, threshold in enumerate(thresholds):
      plt.annotate(f'Threshold {threshold:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(0,10), ha='center')

    
    plt.show()
    return
