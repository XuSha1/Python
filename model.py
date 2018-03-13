#encoding=utf-8
import pandas as pd
import numpy as np
from time import time
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy.stats import randint as sp_randint
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from mlxtend.classifier import StackingClassifier
def RPO_LR(X_train,y_train):
    pipe_lr=Pipeline([('sc',StandardScaler()),
                    ('clf',LogisticRegression(random_state=1))
                     ])# ('pca',PCA(n_components=2,copy=True,whiten=False)
    param_range=[10**c for c in range(-4,4)]
    param_dist={
    'clf__solver':['liblinear','lbfgs','newton-cg','sag','saga'],
    'clf__C':param_range
    }
    n_iter_search=8
    random_rearch=RandomizedSearchCV(pipe_lr,
                                  param_distributions=param_dist,
                                  n_iter=n_iter_search)
    start=time()
    random_rearch.fit(X_train,y_train)
    print("RandomizedSearchCV took %.2f second"%((time()-start)))
    report(random_rearch.cv_results_)
    print(random_rearch.best_score_)
    print(confusion_matrix(y_test,random_rearch.predict(X_test)))
    print(classification_report(y_test,random_rearch.predict(X_test),digits=3))
    return(random_rearch.best_estimator_)
##RandomForest
def RPO_RF(X_train,y_train):
    RF=RandomForestClassifier(n_estimators=20,class_weight='balanced')
    param_dist={
   #     "pca__n_components":sp_randint(1,16),
        "max_depth":[16,None],
        "max_features":sp_randint(1,17),
        "min_samples_split":sp_randint(2,17),
        "min_samples_leaf":sp_randint(1,17),
        "bootstrap":[True,False],
        "criterion":["gini","entropy"]
    }

    n_iter_search=20
    random_search=RandomizedSearchCV(RF,
                                    param_distributions=param_dist,
                                    n_iter=n_iter_search )
    start=time()
    random_search.fit(X_train,y_train)
    print("RandomizedSearchCV took %.2f second" % ((time() - start)))
    report(random_search.cv_results_)
    print(confusion_matrix(y_test, random_search.predict(X_test)))
    print(classification_report(y_test, random_search.predict(X_test),digits=3))
    return(random_search.cv_results_)
def RPO_XGBoost(X_train,y_train):
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=140,
        max_depth=9,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    param_test1 = {
            'max_depth': range(3, 10, 2),
            'min_child_weight': range(1, 6, 2)
        }
    param_test3={
        'gamma':[i/10.0 for i in range(0,5)]
    }

    param_test4={
           'subsample':[i/10.0 for i in range(6,10)],
           'colsample_bytree':[i/10.0 for i in range(6,10)]
    }
    param_test6={
        'reg_alpha':[1e-5,1e-2,0.1,1,100]
    }
    n_iter_search=7
    random_search=RandomizedSearchCV(xgb1,
                     param_distributions=param_test1,
                     n_iter=n_iter_search,
                     cv=5)
    start=time()
    random_search.fit(X_train,y_train)
    print("RandomizedSearchCV took %.2f second" % ((time() - start)))
    report(random_search.cv_results_)
    print(confusion_matrix(Test_target, random_search.predict(Test_target)))
    print(classification_report(Test_target, random_search.predict(Test_target),digits=3))
    return(random_search.cv_results_)
def KNN(X_train,y_train):
    knn=KNeighborsClassifier(n_neighbors=10,weights='distance')
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
def stacking(X, y):
    clf1 = KNeighborsClassifier(n_neighbors=10, weights='distance')
    # clf2 = RandomForestClassifier(n_estimators=20,min_samples_leaf=1, criterion='gini',max_features=8, max_depth=None, bootstrap=False,min_samples_split=8)
    clf2 = RandomForestClassifier(n_estimators=20, min_samples_leaf=4, criterion='entropy', max_features=7,
                                  max_depth=None, bootstrap=False, min_samples_split=9, class_weight='balanced')
    clf3 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=140,
        max_depth=9,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27,

    )
    lr = LogisticRegression(random_state=1, solver='liblinear', C=0.1)
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              meta_classifier=lr)

    print('3-fold cross validation:\n')
    for clf, label in zip([clf1, clf2, clf3, sclf],
                          ['KNN',
                           'Random Forest',
                           'Xgboost',
                           'StackingClassifier']):
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        print('Accuracy:%0.2f (+/- %0.2f)[%s]'
              % (scores.mean(), scores.std(), label))
        clf.fit(X, y)
        print(confusion_matrix(Test_target, clf.predict(Test_data)))
        print(classification_report(Test_target, clf.predict(Test_data)))
def report(results,n_top=3):
    for i in range(1,n_top+1):
        candidates=np.flatnonzero(results['rank_test_score']==i)
        for candidate in candidates:
            print("Model with rank:{0}".format(i))
            print("Mean validation score:{0:.3f}(std:{1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate] ))
            print("Pacameters:{0}".format(results['params'][candidate]))
            print("")
def ReadData(file_name):
    with open(file_name, 'r') as f:
        data = []
        for line in f.readlines():
            line = line.replace('[', '')
            line = line.replace(']', '')
            data.append(eval(line))
        df = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',\
                                         'x9', 'x10','x11', 'x12', 'x13', 'x14', 'x15','x16'])
    return df
if __name__=='__main__':
    lis=['G:/geetest/gt-kaggle/train_data/train_people/feature_355808/part-00000',\
         'G:/geetest/gt-kaggle/train_data/train_robot/feature_93900/part-00000',\
         'G:/geetest/gt-kaggle/test_data/test_people/feature_43955/part-00000',\
         'G:/geetest/gt-kaggle/test_data/test_robot_c/feature_37702/part-00000',\
         'G:/geetest/gt-kaggle/test_data/test_robot_Cl/feature_17139/part-00000']
    train_people=ReadData(lis[0])##355808
    train_robot=ReadData(lis[1])##93900
    test_people=ReadData(lis[2])
    test_robot_c=ReadData(lis[3])
    test_robot_CI=ReadData(lis[4])
    # train_people=train_people.sample(n=93900,replace=True)  ##people_sam=df1.sample(n=1000,replace=True)
    smote_data=pd.read_csv('G:/geetest/smote_data01.csv',sep=',')
    train_robot=pd.concat([train_robot,smote_data],axis=0,ignore_index=True)
    X = pd.concat([train_people, train_robot], axis=0, ignore_index=True) ##axis=0按列合并
    y = pd.DataFrame(np.vstack((np.zeros((355808, 1)),
                                 np.ones((375600, 1)))), columns=['label'])  # 0标记为人，1标记为机器
    X=np.array(X)
    y=np.array(y).reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20, random_state=1)
    ###测试数据
    Test_data=pd.concat([test_people,test_robot_c,test_robot_CI],axis=0,ignore_index=True)
    Test_target=pd.DataFrame(np.vstack((np.zeros((43955,1)),np.ones((54841,1)))),columns=['label'])


    RPO_LR(X_train, y_train)
    #RPO_RF(X_train,y_train.ravel())###0.894
    #RPO_XGBoost(X_train,y_train)
    #KNN(X_train,y_train)
    #stacking(X_train,y_train)






