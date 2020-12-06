import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import PassiveAggressiveRegressor, BayesianRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from Preprocessing.DataObject import DataObject

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
	def __init__(self, models):
		self.models = models
		
	def fit(self, X, y):
		self.models_ = [clone(x) for x in self.models]
		for model in self.models_:
			model.fit(X, y)
		return self
	
	def predict(self, X):
		predictions = np.column_stack([model.predict(X) for model in self.models_])
		return np.mean(predictions, axis=1)   

class select_fetaures(object): 
	def __init__(self, select_cols):
		self.select_cols_ = select_cols
	
	def fit(self, X, Y ):
		return self

	def transform(self, X):
		return X.loc[:, self.select_cols_]    

	def fit_transform(self, X, Y):
		self.fit(X, Y)
		df = self.transform(X)
		return df 

	def __getitem__(self, x):
		return self.X[x], self.Y[x]

class Modeling:
	def __init__(self, dataObject):
		self.trainingData = dataObject.trainingData
		self.testingData = dataObject.testingData
		self.combinedData = dataObject.combinedData
		self.randomState = 183
	
	def go(self, all_data, totalCols, test_ID, colsP, RFEcv, XGBestCols):
		train = all_data.loc[all_data.SalePrice>0 , list(totalCols)].reset_index(drop=True, inplace=False)
		y_train = all_data.SalePrice[all_data.SalePrice>0].reset_index(drop=True, inplace=False)
		test = all_data.loc[all_data.SalePrice==0 , list(totalCols)].reset_index(drop=True, inplace=False)

		scale = RobustScaler() 
		df = scale.fit_transform(train)

		pca = PCA().fit(df) # whiten=True
		print('With only 120 features: {:6.4%}'.format(sum(pca.explained_variance_ratio_[:120])),"%\n")

		print('After PCA, {:3} features only not explained {:6.4%} of variance ratio from the original {:3}'.format(120,
																							(sum(pca.explained_variance_ratio_[120:])),
																							df.shape[1]))
		

		y_train = np.expm1(y_train)

		#Common parameters
		unionedColumns = list(set(RFEcv).union(set(colsP)))
		lengthOfUnionedColumns = len(unionedColumns)

		#XGBRegressor
		model = Pipeline([('pca', PCA(random_state = self.randomState)), ('model', XGBRegressor(random_state = self.randomState, silent=True))])
		gridSearch = self.createGridSearch(model, "XGB", lengthOfUnionedColumns)
		xgbRegressor = Pipeline([('sel', select_fetaures(select_cols = unionedColumns)), ('scl', RobustScaler()), ('gs', gridSearch)])
		xgbRegressor.fit(train, y_train)


		#bayesian ridge
		model = Pipeline([('pca', PCA(random_state = self.randomState)), ('model', BayesianRidge())])
		gridSearch = self.createGridSearch(model, "Bayesian", lengthOfUnionedColumns)
		bayesianRidge = Pipeline([('sel', select_fetaures(select_cols = unionedColumns)), ('scl', RobustScaler()), ('gs', gridSearch)])
		bayesianRidge.fit(train, y_train)

		#Passive Aggressive Regressor
		model = Pipeline([('pca', PCA(random_state = self.randomState)), ('model', PassiveAggressiveRegressor(random_state = self.randomState))])
		gridSearch = self.createGridSearch(model, "PassiveAggressive", lengthOfUnionedColumns)
		passiveAggressiveRegressor = Pipeline([('sel', select_fetaures(select_cols = unionedColumns)), ('scl', RobustScaler()), ('gs', gridSearch)])
		passiveAggressiveRegressor.fit(train, y_train)
		
		averagingModels = AveragingModels(models = (xgbRegressor, bayesianRidge, passiveAggressiveRegressor))
		averagingModels.fit(train, y_train) 
		averagedModelTrainingDataPredictions = averagingModels.predict(train)
		averagedModelTestDataPredictions = (averagingModels.predict(test))
		meanSquaredError = (np.sqrt(mean_squared_error(y_train, averagedModelTrainingDataPredictions)))
		averageModelScore = averagingModels.score(train, y_train)
		
		print('RMSLE score on the train data: {:.4f}'.format(meanSquaredError))
		print('Accuracy score: {:.6%}'.format(averageModelScore))
		
		ensemble = averagedModelTestDataPredictions *1
		submit = pd.DataFrame()
		submit['id'] = test_ID
		submit['SalePrice'] = ensemble
		
		return(submit)

	def createGridSearch(self, model, modelType, lengthOfUnionedColumns):
		if modelType == "XGB":
			parameters = dict(pca__n_components = [90], pca__whiten = [True], model__n_estimators = [3500], 
				model__booster = ['gblinear'], model__objective = ['reg:tweedie'], model__learning_rate = [0.01],
				model__reg_lambda = [1], model__reg_alpha = [1], model__max_depth = [3]) 
		elif modelType == "Bayesian":
			parameters = dict(pca__n_components = [lengthOfUnionedColumns - 9], pca__whiten = [True], model__n_iter = [36],
				model__alpha_1 = [1e-06], model__alpha_2 = [0.1], model__lambda_1 = [0.001], model__lambda_2 = [0.01]) 
		elif modelType == "PassiveAggressive":
			n_components = [lengthOfUnionedColumns-9, lengthOfUnionedColumns-8, lengthOfUnionedColumns-7, lengthOfUnionedColumns-1] 
			parameters = dict(pca__n_components = n_components, pca__whiten = [True], model__loss = ['squared_epsilon_insensitive'],
				model__epsilon = [0.00001], model__C = [0.001], model__tol = [0.001], model__max_iter = [1000])

		gridSearchScoringList = list(['neg_mean_squared_error' , 'neg_mean_absolute_error', 'r2'])
		gridSearchRefitFunction = 'neg_mean_squared_error'

		return GridSearchCV(estimator = model, param_grid = parameters, refit = gridSearchRefitFunction, 
			scoring = gridSearchScoringList, cv = 5, n_jobs = 4)

