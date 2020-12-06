import pandas as pd
import random as rnd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from Preprocessing.PreliminaryDataAdjuster import PreliminaryDataAdjuster
from Preprocessing.OrdinalToNumericalConverter import OrdinalToNumericalConverter
from Preprocessing.DataObject import DataObject
from Preprocessing.FeatureEngineering import FeatureEngineering
from Preprocessing.SelectFeatures import SelectFeatures
from Preprocessing.Modeling import Modeling

# main class for processing data
class Preprocessor:
	def __init__(self, trainingData, testingData):
		self.trainingData = trainingData
		self.testingData = testingData
		self.combinedData = [trainingData, testingData]

	# main function that combines all preprocessing.
	def process(self, test_ID):
		dataObject = DataObject(self.trainingData, self.testingData, self.combinedData)

		PDA = PreliminaryDataAdjuster(dataObject)
		dataObject = PDA.go()

		ONC = OrdinalToNumericalConverter(dataObject)
		dataObject = ONC.go()

		FE = FeatureEngineering(dataObject)
		dataObject, combinedData, y_train, cols, colsP = FE.go()

		SF = SelectFeatures(dataObject)
		dataObject, totalCols, RFEcv, XGBestCols = SF.go(combinedData, cols, colsP)
		
		model = Modeling(dataObject)
		ouput_ensembled = model.go(combinedData, totalCols, test_ID, colsP, RFEcv, XGBestCols)


		ouput_ensembled.to_csv('SalePrice_N_submission.csv', index = False)

		print(dataObject.trainingData)

		return dataObject
