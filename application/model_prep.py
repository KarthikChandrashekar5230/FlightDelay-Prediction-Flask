from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import numpy as np
import pandas as pd
import pickle


class ModelBuilding:

    def randonmizedgridsearchCV(self,input_train,target_train):

        n_estimators = [int(x) for x in np.linspace(start = 1, stop = 1000, num = 150)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(1, 15, num = 1)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10,12]
        min_samples_leaf = [1, 2, 4,6]
        bootstrap = [True, False]
        #cv = ShuffleSplit(n_splits=2, test_size=0.1)

        random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions= random_grid, n_iter = 25, cv = 3,
                               verbose=2, n_jobs = -1)
        rf_random.fit(input_train, target_train)

        print("Best Estimator suited for the model: ",rf_random.best_estimator_)
        print("Best parameters suited for the model: ",rf_random.best_params_)
        results=pd.DataFrame(rf_random.cv_results_)

        return None

    def randonmizedgridsearchCV_results(self,filepath):

        randomizedsearchcv_results = pd.read_csv("filepath", header=0)
        print(randomizedsearchcv_results)

        return None

    def modelbuilding_performance(self,input_train, input_test, target_train, target_test):

        model_rf = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse', max_depth=None,
                                         max_features='auto',
                                         max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0,
                                         min_impurity_split=None,
                                         min_samples_leaf=2, min_samples_split=10, min_weight_fraction_leaf=0.0,
                                         n_estimators=852,
                                         n_jobs=-1, oob_score=True, random_state=13, verbose=0, warm_start=False)

        model_rf.fit(input_train, target_train)
        pred_train = model_rf.predict(input_train)
        pred_test = model_rf.predict(input_test)

        #pickle.dump(model_rf,open("RFRegressor.pkl",'wb'))

        print("RandomForest Classifier performance based on OOB_Score is: ", model_rf.oob_score_)
        print('Root Mean Squared Error for Training:', np.sqrt(metrics.mean_squared_error(pred_train, target_train)))
        print('Root Mean Squared Error for Test:', np.sqrt(metrics.mean_squared_error(pred_test, target_test)))

        print('Training score by RF_Regressor:',model_rf.score(input_train, target_train))
        print('Testing score by RF_Regressor:',model_rf.score(input_test, target_test))

        return None


