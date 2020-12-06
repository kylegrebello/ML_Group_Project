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
from sklearn.base import TransformerMixin
from Utils import Utils

from Preprocessing.DataObject import DataObject


# Removes outliers and unnecessary columns and fills n/a values.
class PreliminaryDataAdjuster:
	def __init__(self, dataObject):
		self.trainingData = dataObject.trainingData
		self.testingData = dataObject.testingData
		self.combinedData = dataObject.combinedData

	def go(self):
		self.trainingData.drop("Id", axis = 1, inplace = True)
		self.testingData.drop("Id", axis = 1, inplace = True)

		self.trainingData.rename(columns={'3SsnPorch':'TSsnPorch'}, inplace=True)
		self.testingData.rename(columns={'3SsnPorch':'TSsnPorch'}, inplace=True)

		self.testingData['SalePrice'] = 0

		self.trainingData = self.trainingData.drop(
			self.trainingData[(self.trainingData.GrLivArea > 4000) & (self.trainingData.SalePrice < 300000)].index)

		self.trainingData = self.trainingData[self.trainingData.GrLivArea * self.trainingData.TotRmsAbvGrd < 45000]

		self.trainingData = self.trainingData[self.trainingData.GarageArea * self.trainingData.GarageCars < 3700]

		self.trainingData = self.trainingData[(self.trainingData.FullBath + (
					self.trainingData.HalfBath * 0.5) + self.trainingData.BsmtFullBath + (
														   self.trainingData.BsmtHalfBath * 0.5)) < 5]

		self.trainingData = self.trainingData.loc[~(self.trainingData.SalePrice == 392500.0)]
		self.trainingData = self.trainingData.loc[
			~((self.trainingData.SalePrice == 275000.0) & (self.trainingData.Neighborhood == 'Crawfor'))]
		self.trainingData.SalePrice = np.log1p(self.trainingData.SalePrice)

		CDQ = CheckDataQuality(self.trainingData, self.testingData)
		self.trainingData, self.testingData = CDQ.go()

		self.combinedData = [self.trainingData, self.testingData]

		return DataObject(self.trainingData, self.testingData, self.combinedData)

	def fillMissingData(self, dataset):
		labelsToFillWithNA = ['Alley', 'Fence', 'MiscFeature', 'PoolQC', 'FireplaceQu']

		dataset = self.fillMSZoningMissingValues(dataset)
		dataset = self.fillLotFrontageMissingValues(dataset)
		dataset = self.fillMasonryVeneerMissingValues(dataset)
		dataset = self.fillExteriorCoveringMissingValues(dataset)
		dataset = self.fillBasementFeaturesMissingValues(dataset)
		dataset = self.fillElectricalMissingValues(dataset)
		dataset = self.fillKitchenQualityMissingValues(dataset)
		dataset = self.fillGarageFeaturesMissingValues(dataset)
		dataset = self.fillPoolQualityMissingValues(dataset)
		dataset = self.fillSaleTypeMissingValues(dataset)

		dataset = Utils.fillNullLabels(dataset, labelsToFillWithNA, 'NA')
		dataset = Utils.fillNullLabels(dataset, ['Functional'],
									   'Typ')

		return dataset

	# This function handles the MSZoning missing values
	# Since missing values are small for this feature we will just fill with the most
	#    common value in the dataset
	def fillMSZoningMissingValues(self, dataset):
		mostFrequentZoningValue = dataset.MSZoning.dropna().mode()[0]
		dataset['MSZoning'] = dataset['MSZoning'].fillna(mostFrequentZoningValue)

		return dataset

	# This function handles the Lot Frontage Features missing values
	# We are going to first group the Lot Frontage Features by neighborhood
	# Then we are going to fill the missing values with the mean Lot Frontage in the neighborhood
	#
	# We are grouping by neighborhood because houses in a neighborhood have similiar LotFrontage values
	def fillLotFrontageMissingValues(self, dataset):
		neighborhoodLotFrontageMeans = dataset.groupby('Neighborhood').LotFrontage.mean()
		lotFrontageValues = (dataset.loc[dataset.LotFrontage.isnull(), ['Neighborhood']]).transpose()

		lotFrontageFeature = dataset['LotFrontage'].copy()
		for i in lotFrontageValues:
			lotFrontageFeature[lotFrontageValues[i].name] = neighborhoodLotFrontageMeans[lotFrontageValues[i].values[0]]

		dataset['LotFrontage'] = lotFrontageFeature
		return dataset

	def fillMasonryVeneerMissingValues(self, dataset):
		masonryVeneerCase1NULL = (
			dataset.loc[(dataset.MasVnrType.isnull()) & (dataset.MasVnrArea > 0), ['MasVnrType']]).transpose()
		masonryVeneerCase1None = (
			dataset.loc[(dataset.MasVnrType == 'None') & (dataset.MasVnrArea > 0), ['MasVnrType']]).transpose()
		masonryVeneerCase1 = masonryVeneerCase1NULL.append(masonryVeneerCase1None)
		masonryVeneerCase2 = (
			dataset.loc[(dataset.MasVnrType != 'None') & (dataset.MasVnrArea == 0), ['MasVnrArea']]).transpose()
		medianOfMasonryVeneerCase2 = \
			dataset.loc[(dataset.MasVnrType != 'None') & (dataset.MasVnrArea > 0), ['MasVnrArea']].median()[0]

		mostCommon = dataset['MasVnrType'].value_counts().index[0]
		if (mostCommon == 'None'):
			mostCommon = dataset['MasVnrType'].value_counts().index[1]

		masVnrTypeFeature = dataset['MasVnrType'].copy()
		for i in masonryVeneerCase1:
			masVnrTypeFeature[masonryVeneerCase1[i].name] = mostCommon

		masVnrAreaFeature = dataset['MasVnrArea'].copy()
		for i in masonryVeneerCase2:
			masVnrAreaFeature[masonryVeneerCase2[i].name] = medianOfMasonryVeneerCase2

		dataset['MasVnrType'] = masVnrTypeFeature
		dataset['MasVnrArea'] = masVnrAreaFeature

		dataset['MasVnrType'] = dataset['MasVnrType'].fillna('None')
		dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)
		return dataset

	# This function handles the Garage Features missing values
	# Need to handle the following cases:
	#    1. Fill GarageType NULL values with 'NA'
	#    2. Handle case where GarageType is Detchd but the rest of the row is NULL
	def fillGarageFeaturesMissingValues(self, dataset):
		dataset = Utils.fillNullLabels(dataset, ['GarageType'], 'NA')

		dataset = self.fillGarageFeatureValue(dataset, 'GarageYrBlt', 'median')
		dataset = self.fillGarageFeatureValue(dataset, 'GarageFinish', 'mode')
		dataset = self.fillGarageFeatureValue(dataset, 'GarageCars', 'median')
		dataset = self.fillGarageFeatureValue(dataset, 'GarageArea', 'median')
		dataset = self.fillGarageFeatureValue(dataset, 'GarageQual', 'mode')
		dataset = self.fillGarageFeatureValue(dataset, 'GarageCond', 'mode')

		return dataset

	def fillGarageFeatureValue(self, dataset, feature, fillType):
		if (fillType == 'median'):
			fillValue = dataset[dataset.GarageType == 'Detchd'][feature].median()
			fillnaValue = 0
		elif (fillType == 'mode'):
			fillValue = dataset[dataset.GarageType == 'Detchd'][feature].mode()[0]
			fillnaValue = 'NA'

		condition = (dataset.GarageType == 'Detchd') & (dataset[feature].isnull())
		values = (dataset.loc[condition, [feature]]).transpose()

		datasetFeature = dataset[feature].copy()
		for i in values:
			datasetFeature[values[i].name] = fillValue

		dataset[feature] = datasetFeature
		dataset[feature] = dataset[feature].fillna(fillnaValue)
		return dataset

	# This function handles the cases where the PoolArea is greater than zero but PoolQC is NULL
	# We are going to use the OverallQuality feature of the house to determine the pool quality
	# The Pool quality has 5 categorical features and OverallQuality has 10 categorical features.
	#   We will divide OverallQuality by 2 to get the correlated pool quality value
	def fillPoolQualityMissingValues(self, dataset):
		poolQualityMap = {0: 'NA', 1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'}
		poolQuality = (
			(dataset.loc[(dataset.PoolArea > 0) & (dataset.PoolQC.isnull()), ['OverallQual']] / 2).round()).transpose()

		poolQCFeature = dataset['PoolQC'].copy()
		for i in poolQuality:
			poolQCFeature[i] = poolQuality[i].map(poolQualityMap)

		dataset['PoolQC'] = poolQCFeature
		return dataset

	# This function handles the cases where kitchen quality is missing
	# Currently there is only one value missing and it is in the testing set,
	#   so we will just fill with the most common value in the dataset
	def fillKitchenQualityMissingValues(self, dataset):
		kitchenQualityMode = dataset.KitchenQual.mode()[0]
		kitchenQuality = (
			dataset.loc[(dataset.KitchenAbvGr > 0) & (dataset.KitchenQual.isnull()), ['KitchenQual']]).transpose()

		kitchenQualFeature = dataset['KitchenQual'].copy()
		for i in kitchenQuality:
			kitchenQualFeature[kitchenQuality[i].name] = kitchenQualityMode

		dataset['KitchenQual'] = kitchenQualFeature
		return dataset

	def fillSaleTypeMissingValues(self, dataset):
		mode = dataset['SaleType'].mode()[0]
		dataset['SaleType'] = dataset['SaleType'].fillna(mode)
		return dataset

	# This function is called in 'fillBasementFeaturesMissingValues'
	# It is for setting basement features with most common when they
	#    are equal to null but the basement has a non zero area
	def fillBasementFeatureWithMostCommon(self, dataset, feature, condition):
		mostCommon = dataset[feature].value_counts().index[0]
		if (mostCommon == 'No'):
			mostCommon = dataset[feature].value_counts().index[1]

		values = (dataset.loc[condition, [feature]]).transpose()

		datasetFeature = dataset[feature].copy()
		for i in values:
			datasetFeature[values[i].name] = mostCommon

		dataset[feature] = datasetFeature
		return dataset

	# This function handles the Basement Features missing values
	# Need to handle the following cases:
	#    1. TotalBsmtSF is greater than zero but BsmtExposure is NULL
	#    2. TotalBsmtSF is greater than zero but BsmtQual is NULL
	#    3. TotalBsmtSF is greater than zero but BsmtCond is NULL
	#    4. BsmtFinSF2 is greater than zero but BsmtFinType2 is NULL
	#    5. BsmtFinSF2 is zero and BsmtUnfSF is not zero but BsmtFinType2 is finished
	#       - Set BsmtFinSF2 = BsmtUnfSF and set BsmtUnfSF = 0
	#    6. Fill categorical basement data with NA
	#    7. Fill numerical basement data with 0
	def fillBasementFeaturesMissingValues(self, dataset):
		bsmtQualCondition = (dataset.TotalBsmtSF > 0) & (dataset.BsmtQual.isnull())
		bsmtCondCondition = (dataset.TotalBsmtSF > 0) & (dataset.BsmtCond.isnull())
		bsmtExposureCondition = (dataset.TotalBsmtSF > 0) & (dataset.BsmtExposure.isnull())
		bsmtFinType2Condition = (dataset.BsmtFinSF2 > 0) & (dataset.BsmtFinType2.isnull())
		bsmtFinSF2Condition = (dataset.BsmtFinSF2 == 0) & (dataset.BsmtFinType2 != 'Unf') & (
			~dataset.BsmtFinType2.isnull())

		dataset = self.fillBasementFeatureWithMostCommon(dataset, 'BsmtQual', bsmtQualCondition)
		dataset = self.fillBasementFeatureWithMostCommon(dataset, 'BsmtCond', bsmtCondCondition)
		dataset = self.fillBasementFeatureWithMostCommon(dataset, 'BsmtExposure', bsmtExposureCondition)
		dataset = self.fillBasementFeatureWithMostCommon(dataset, 'BsmtFinType2', bsmtFinType2Condition)
		dataset = self.handleBsmtFinSF2SpecialCase(dataset, bsmtFinSF2Condition)

		basementFeaturesToFillWithNA = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
		basementFeaturesToFillWith0 = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
									   'BsmtHalfBath']
		dataset = Utils.fillNullLabels(dataset, basementFeaturesToFillWithNA, 'NA')
		dataset = Utils.fillNullLabels(dataset, basementFeaturesToFillWith0, 0)

		return dataset

	# This function handles the missing values of Electrical
	# Since there are very few missing values we will fill with the most common
	def fillElectricalMissingValues(self, dataset):
		mostCommon = dataset['Electrical'].value_counts().index[0]
		dataset['Electrical'] = dataset['Electrical'].fillna(mostCommon)

		return dataset

	def fillExteriorCoveringMissingValues(self, dataset):
		exterior1stMode = dataset['Exterior1st'].mode()[0]
		exterior2ndMode = dataset['Exterior2nd'].mode()[0]
		dataset['Exterior1st'] = dataset['Exterior1st'].fillna(exterior1stMode)
		dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(exterior2ndMode)

		return dataset



	# This function is called in 'fillBasementFeaturesMissingValues'
	# It handles the case where BsmtFinSF2 is zero and BsmtUnfSF is zero,
	#   but BsmtFinType2 is finished
	def handleBsmtFinSF2SpecialCase(self, dataset, condition):
		values = (dataset.loc[condition, ['BsmtFinSF2', 'BsmtUnfSF']]).transpose()

		bsmtFinSF2Feature = dataset['BsmtFinSF2'].copy()
		bsmtUnfSFFeature = dataset['BsmtUnfSF'].copy()
		for i in values:
			currentUnfSFValue = values[i].values[1]
			bsmtFinSF2Feature[values[i].name] = currentUnfSFValue
			bsmtUnfSFFeature[values[i].name] = 0

		dataset['BsmtFinSF2'] = bsmtFinSF2Feature
		dataset['BsmtUnfSF'] = bsmtUnfSFFeature
		return dataset


# Currently unused
class DT(TransformerMixin):

	def fit(self, X, y=None):

		self.fill = pd.Series([X[c].value_counts().index[0]
			if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
			index=X.columns)

		return self

	def transform(self, X, y=None):
		return X.fillna(self.fill)


class CheckDataQuality:
	def __init__(self, trainingData, testingData):
		self.trainingData = trainingData
		self.testingData = testingData

	def go(self):
		all_data = pd.concat((self.trainingData, self.testingData)).reset_index(drop=True)
		all_data.drop('Utilities', axis=1, inplace=True)
		all_data.Electrical = all_data.Electrical.fillna('SBrkr')

		all_data.GarageType = all_data.GarageType.fillna('NA')

		# Group by GarageType and fill missing value with median where GarageType=='Detchd' and 0 for the others
		cmedian = all_data[all_data.GarageType=='Detchd'].GarageArea.median()
		all_data.loc[all_data.GarageType=='Detchd', 'GarageArea'] = all_data.loc[all_data.GarageType=='Detchd',
																				 'GarageArea'].fillna(cmedian)
		all_data.GarageArea = all_data.GarageArea.fillna(0)

		cmedian = all_data[all_data.GarageType=='Detchd'].GarageCars.median()
		all_data.loc[all_data.GarageType=='Detchd', 'GarageCars'] = all_data.loc[all_data.GarageType=='Detchd', 'GarageCars'].fillna(cmedian)
		all_data.GarageCars = all_data.GarageCars.fillna(0)

		cmedian = all_data[all_data.GarageType=='Detchd'].GarageYrBlt.median()
		all_data.loc[all_data.GarageType=='Detchd', 'GarageYrBlt'] = all_data.loc[all_data.GarageType=='Detchd', 'GarageYrBlt'].fillna(cmedian)
		all_data.GarageYrBlt = all_data.GarageYrBlt.fillna(0)

		# Group by GarageType and fill missing value with mode where GarageType=='Detchd' and 'NA' for the others
		cmode = all_data[all_data.GarageType=='Detchd'].GarageFinish.mode()[0]
		all_data.loc[all_data.GarageType=='Detchd', 'GarageFinish'] = all_data.loc[all_data.GarageType=='Detchd', 'GarageFinish'].fillna(cmode)
		all_data.GarageFinish = all_data.GarageFinish.fillna('NA')

		cmode = all_data[all_data.GarageType=='Detchd'].GarageQual.mode()[0]
		all_data.loc[all_data.GarageType=='Detchd', 'GarageQual'] = all_data.loc[all_data.GarageType=='Detchd', 'GarageQual'].fillna(cmode)
		all_data.GarageQual = all_data.GarageQual.fillna('NA')

		cmode = all_data[all_data.GarageType=='Detchd'].GarageCond.mode()[0]
		all_data.loc[all_data.GarageType=='Detchd', 'GarageCond'] = all_data.loc[all_data.GarageType=='Detchd', 'GarageCond'].fillna(cmode)
		all_data.GarageCond = all_data.GarageCond.fillna('NA')

		all_data.loc[(all_data.MasVnrType=='None') & (all_data.MasVnrArea>0), ['MasVnrType']] = 'BrkFace'

		# All Types null with Are greater than 0 update to BrkFace type
		all_data.loc[(all_data.MasVnrType.isnull()) & (all_data.MasVnrArea>0), ['MasVnrType']] = 'BrkFace'

		# All Types different from None with Are equal to 0 update to median Area of no None types with Areas
		all_data.loc[(all_data.MasVnrType!='None') & (all_data.MasVnrArea==0), ['MasVnrArea']] = all_data.loc[(all_data.MasVnrType!='None') & (all_data.MasVnrArea>0), ['MasVnrArea']].median()[0]
		# Filling 0 and None for records wheres both are nulls
		all_data.MasVnrArea = all_data.MasVnrArea.fillna(0)
		all_data.MasVnrType = all_data.MasVnrType.fillna('None')

		all_data.loc[(~all_data.TotalBsmtSF.isnull()) & (all_data.BsmtExposure.isnull()) & (all_data.TotalBsmtSF>0), 'BsmtExposure'] = 'Av'
		all_data.loc[(~all_data.TotalBsmtSF.isnull()) & (all_data.BsmtQual.isnull()) & (all_data.TotalBsmtSF>0), 'BsmtQual'] = 'TA'
		all_data.loc[(~all_data.TotalBsmtSF.isnull()) & (all_data.BsmtCond.isnull()) & (all_data.TotalBsmtSF>0), 'BsmtCond'] = 'TA'
		all_data.loc[(all_data.BsmtFinSF2>0) & (all_data.BsmtFinType2.isnull()) , 'BsmtFinType2'] = 'Unf'
		all_data.loc[(all_data.BsmtFinSF2==0) & (all_data.BsmtFinType2!='Unf') & (~all_data.BsmtFinType2.isnull()), 'BsmtFinSF2'] = 354.0
		all_data.loc[(all_data.BsmtFinSF2==0) & (all_data.BsmtFinType2!='Unf') & (~all_data.BsmtFinType2.isnull()), 'BsmtUnfSF'] = 0.0

		nulls_cols = {'BsmtExposure': 'NA', 'BsmtFinType2': 'NA', 'BsmtQual': 'NA', 'BsmtCond': 'NA', 'BsmtFinType1': 'NA',
					  'BsmtFinSF1': 0, 'BsmtFinSF2': 0, 'BsmtUnfSF': 0 ,'TotalBsmtSF': 0, 'BsmtFullBath': 0, 'BsmtHalfBath': 0}

		all_data = all_data.fillna(value=nulls_cols)

		NegMean = all_data.groupby('Neighborhood').LotFrontage.mean()

		all_data.loc.LotFrontage = all_data[['Neighborhood', 'LotFrontage']].apply(lambda x: NegMean[x.Neighborhood] if np.isnan(x.LotFrontage) else x.LotFrontage, axis=1)

		PoolQC = {0: 'NA', 1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'}

		all_data.loc[(all_data.PoolArea>0) & (all_data.PoolQC.isnull()), ['PoolQC']] =\
				((all_data.loc[(all_data.PoolArea>0) & (all_data.PoolQC.isnull()), ['OverallQual']]/2).round()).\
				apply(lambda x: x.map(PoolQC))

		all_data.PoolQC = all_data.PoolQC.fillna('NA')

		all_data.Functional = all_data.Functional.fillna('Typ')

		all_data.loc[(all_data.Fireplaces==0) & (all_data.FireplaceQu.isnull()), ['FireplaceQu']] = 'NA'

		all_data.loc[(all_data.KitchenAbvGr>0) & (all_data.KitchenQual.isnull()),
					 ['KitchenQual']] = all_data.KitchenQual.mode()[0]

		all_data.Alley = all_data.Alley.fillna('NA')
		all_data.Fence = all_data.Fence.fillna('NA')
		all_data.MiscFeature = all_data.MiscFeature.fillna('NA')
		all_data.loc[all_data.GarageYrBlt==2207.0, 'GarageYrBlt'] = 2007.0

		all_data = DT().fit_transform(all_data)

		self.trainingData = all_data.loc[(all_data.SalePrice>0)].reset_index(drop=True, inplace=False)
		self.testingData = all_data.loc[(all_data.SalePrice==0)].reset_index(drop=True, inplace=False)

		return self.trainingData, self.testingData
