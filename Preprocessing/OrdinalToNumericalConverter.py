import pandas as pd
import random as rnd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from Preprocessing.DataObject import DataObject


class OrdinalToNumericalConverter:
	def __init__(self, dataObject):
		self.trainingData = dataObject.trainingData
		self.testingData = dataObject.testingData
		self.combinedData = dataObject.combinedData

	def go(self):
		self.trainingData = self.mapCategoricalToOrdinal(self.trainingData)
		self.testingData = self.mapCategoricalToOrdinal(self.testingData)
		self.combinedData = [self.trainingData, self.testingData]

		return DataObject(self.trainingData, self.testingData, self.combinedData)

	def mapCategoricalToOrdinal(self, dataset):
		ordinal_label = ['LandSlope', 'ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual',
						 'FireplaceQu', 'GarageCond', 'PavedDrive', 'LotShape', 'BsmtQual', 'BsmtCond', 'GarageQual',
						 'PoolQC', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'CentralAir', 'GarageFinish',
						 'Functional',
						 'Street', 'Fence']
		number = LabelEncoder()
		for i in ordinal_label:
			dataset[i] = number.fit_transform(dataset[i].astype('str'))
		return dataset
