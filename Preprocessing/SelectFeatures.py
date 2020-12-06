import pandas as pd
import numpy as np
import statsmodels.api as sm

from itertools import combinations
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.base import clone
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest, RFECV, SelectFromModel

from Preprocessing.DataObject import DataObject

randomStateValue = 173
kBestValue = 80

class SequentialFeatureSelection():
	def __init__(self, estimator, k_features, scoring=r2_score, test_size=0.25, random_state=randomStateValue):
		self.scoring = scoring
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state

	def fit(self, X, y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
		dimension = X_train.shape[1]
		self.indices_ = list(range(dimension))
		self.subsets_ = [self.indices_]
		initialScore = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
		self.scores_ = [initialScore]
		
		while dimension > self.k_features:
			listOfCalculatedScores = []
			listOfIndicesSubsets = []
			for currentIndices in combinations(self.indices_, r = dimension-1):
				currentScore = self._calc_score(X_train, y_train, X_test, y_test, list(currentIndices))
				listOfCalculatedScores.append(currentScore)
				listOfIndicesSubsets.append(list(currentIndices))
				
			maxCalculatedScore = np.argmax(listOfCalculatedScores)
			self.indices_ = listOfIndicesSubsets[maxCalculatedScore]
			self.subsets_.append(self.indices_)
			dimension -= 1
			self.scores_.append(listOfCalculatedScores[maxCalculatedScore])
			
		self.k_score_ = self.scores_[-1]
		return self

	def transform(self, X):
		return X.iloc[:, self.indices_]
	
	def _calc_score(self, X_train, y_train, X_test, y_test, indices):
		self.estimator.fit(X_train.iloc[:, indices], y_train)
		y_pred = self.estimator.predict(X_test.iloc[:, indices])
		return self.scoring(y_test, y_pred)

class SelectFeatures:
	def __init__(self, dataObject):
		self.trainingData = dataObject.trainingData
		self.testingData = dataObject.testingData
		self.combinedData = dataObject.combinedData

	def go(self, all_data, cols, polynomialColumns):
		trainingData = all_data.loc[(all_data.SalePrice > 0), cols].reset_index(drop = True, inplace = False)
		y_train = all_data.SalePrice[all_data.SalePrice > 0].reset_index(drop = True, inplace = False)

		robustScaler = RobustScaler();
		robustScalerDataFrame = pd.DataFrame(robustScaler.fit_transform(trainingData[cols]), columns = cols)

		pValueColumns = cols.values
		pValueColumns = self.backwardElimination(robustScalerDataFrame, y_train, pValueColumns)

		lasso = Lasso(alpha = 0.0005, tol = 0.002)
		recursiveFeatureEliminator = RFECV(estimator = lasso, n_jobs = -1, step = 1, scoring = 'neg_mean_squared_error' , cv = 5)
		recursiveFeatureEliminator.fit(robustScalerDataFrame, y_train)

		recursivelySelectedFeatures = recursiveFeatureEliminator.get_support()
		recursiveFeatureSelectedColumns = cols[recursivelySelectedFeatures]

		r2Score = r2_score
		lasso = Lasso(alpha = 0.0005, tol = 0.002)
		sequentialFeatureSelection = SequentialFeatureSelection(lasso, k_features = 1, scoring = r2Score)
		sequentialFeatureSelection.fit(robustScalerDataFrame, y_train)

		sequentialFeatureSelectionScoreLength = len(sequentialFeatureSelection.scores_)
		sequentialFeatureSelectionScoreCriteria = (sequentialFeatureSelection.scores_==max(sequentialFeatureSelection.scores_))
		arrangedSequentialFeatures = np.arange(0, sequentialFeatureSelectionScoreLength)[sequentialFeatureSelectionScoreCriteria]
		maxSequentialFeatureScore = max(arrangedSequentialFeatures)
		sequentialFeatureSelectionSubsets = list(sequentialFeatureSelection.subsets_[maxSequentialFeatureScore])
		sequentialBackwardSelection = list(robustScalerDataFrame.columns[sequentialFeatureSelectionSubsets])

		kBestSelection = SelectKBest(score_func = f_regression, k = kBestValue)
		kBestSelection.fit(robustScalerDataFrame, y_train)
		select_features_kbest = kBestSelection.get_support()
		kbestWithFRegressionScoringFunction = cols[select_features_kbest]

		kBestSelection = SelectKBest(score_func = mutual_info_regression, k = kBestValue)
		kBestSelection.fit(robustScalerDataFrame, y_train)
		select_features_kbest = kBestSelection.get_support()
		kbestWithMutualInfoRegressionScoringFunction = cols[select_features_kbest]

		X_train, X_test, y, y_test = train_test_split(robustScalerDataFrame, y_train, test_size = 0.30, random_state = randomStateValue)
		model =  XGBRegressor(base_score = 0.5, random_state = randomStateValue, n_jobs = 4, silent = True)
		model.fit(X_train, y)

		bestValue = 1e36
		bestColumns = 31
		my_model = model
		threshold = 0

		for modelThreshold in np.sort(np.unique(model.feature_importances_)):
			selectionsFromModel = SelectFromModel(model, threshold = modelThreshold, prefit = True)
			X_trainSelectedFromModel = selectionsFromModel.transform(X_train)
			modelForSelection = XGBRegressor(base_score = 0.5, random_state = randomStateValue, n_jobs = 4, silent = True)
			modelForSelection.fit(X_trainSelectedFromModel, y)
			X_testSelectedFromModel = selectionsFromModel.transform(X_test)
			y_pred = modelForSelection.predict(X_testSelectedFromModel)
			roundedPredictions = [round(predictedValue) for predictedValue in y_pred]
			meanSquaredErrorValue = mean_squared_error(y_test, roundedPredictions)
			if (bestValue >= meanSquaredErrorValue):
				bestValue = meanSquaredErrorValue
				bestColumns = X_trainSelectedFromModel.shape[1]
				my_model = modelForSelection
				threshold = modelThreshold

		listOfFeatureImportance = [(score, feature) for score, feature in zip(model.feature_importances_, cols)]
		XGBestValues = pd.DataFrame(sorted(sorted(listOfFeatureImportance, reverse = True)[:bestColumns]), columns = ['Score', 'Feature'])
		XGBestColumns = XGBestValues.iloc[:, 1].tolist()

		unionSetOfBestColumns = set(pValueColumns)
		unionSetOfBestColumns = unionSetOfBestColumns.union(set(recursiveFeatureSelectedColumns))
		unionSetOfBestColumns = unionSetOfBestColumns.union(set(kbestWithFRegressionScoringFunction))
		unionSetOfBestColumns = unionSetOfBestColumns.union(set(kbestWithMutualInfoRegressionScoringFunction))
		unionSetOfBestColumns = unionSetOfBestColumns.union(set(XGBestColumns))
		unionSetOfBestColumns = unionSetOfBestColumns.union(set(sequentialBackwardSelection))
		unionSetOfBestColumns = unionSetOfBestColumns.union(set(polynomialColumns))
		unionSetOfBestColumnsList = list(unionSetOfBestColumns)

		return DataObject(self.trainingData, self.testingData, self.combinedData), unionSetOfBestColumnsList, recursiveFeatureSelectedColumns, XGBestColumns

	def backwardElimination(self, x, Y, columns):
		numberOfVariables = x.shape[1]
		for i in range(0, numberOfVariables):
			olsRegressor = sm.OLS(Y, x).fit()
			maxPValueFromRegressor = max(olsRegressor.pvalues)
			if maxPValueFromRegressor > 0.051:
				for j in range(0, numberOfVariables - i):
					if (olsRegressor.pvalues[j].astype(float) == maxPValueFromRegressor):
						columns = np.delete(columns, j)
						x = x.loc[:, columns]

		return columns
	
