import pandas as pd
import random as rnd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn import tree
from patsy import dmatrices
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from scipy.stats import skew, norm, probplot, boxcox

from Preprocessing.DataObject import DataObject


class FeatureEngineering:
	def __init__(self, dataObject):
		self.trainingData = dataObject.trainingData
		self.testingData = dataObject.testingData
		self.combinedData = dataObject.combinedData

	def go(self):
		trainShape = self.trainingData.shape[0]

		combinedData = pd.concat((self.trainingData, self.testingData)).reset_index(drop=True)
		combinedData, y_train, cols, colsP = self.featureEngineer(combinedData, trainShape)

		return DataObject(self.trainingData, self.testingData, self.combinedData), combinedData, y_train, cols, colsP

	def featureEngineer(self, combinedData, trainShape):
		combinedData.loc[(combinedData.PoolArea > 0), ['MiscFeature']] = 'Pool'
		combinedData.loc[(combinedData.PoolArea > 0), ['MiscVal']] = combinedData.loc[(combinedData.PoolArea > 0), ['MiscVal', 'PoolArea']].apply(lambda x: (x.MiscVal + x.PoolArea), axis=1)

		combinedData['TotalExtraPoints'] = combinedData.HeatingQC + combinedData.PoolQC + combinedData.FireplaceQu + combinedData.KitchenQual
		combinedData['TotalPoints'] = (combinedData.ExterQual + combinedData.FireplaceQu + combinedData.GarageQual + combinedData.KitchenQual + 
			combinedData.BsmtQual + combinedData.BsmtExposure + combinedData.BsmtFinType1 + combinedData.PoolQC + combinedData.ExterCond + 
			combinedData.BsmtCond + combinedData.GarageCond + combinedData.OverallCond + combinedData.BsmtFinType2 + combinedData.HeatingQC) + combinedData.OverallQual ** 2

		df = combinedData.loc[(combinedData.SalePrice > 0), ['TotalPoints', 'TotalExtraPoints', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'PoolQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'SalePrice']]

		combinedData['GarageArea_x_Car'] = combinedData.GarageArea * combinedData.GarageCars

		combinedData['TotalBsmtSF_x_Bsm'] = combinedData.TotalBsmtSF * combinedData['1stFlrSF']

		combinedData['ConstructArea'] = (combinedData.TotalBsmtSF + combinedData.WoodDeckSF + combinedData.GrLivArea + combinedData.OpenPorchSF +
			combinedData.TSsnPorch + combinedData.ScreenPorch + combinedData.EnclosedPorch + combinedData.MasVnrArea + combinedData.GarageArea + combinedData.PoolArea)

		combinedData['Garage_Newest'] = combinedData.YearBuilt > combinedData.GarageYrBlt
		combinedData.Garage_Newest = combinedData.Garage_Newest.apply(lambda x: 1 if x else 0)

		combinedData[
			'TotalPorchSF'] = combinedData.OpenPorchSF + combinedData.EnclosedPorch + combinedData.TSsnPorch + combinedData.ScreenPorch + combinedData.WoodDeckSF
		combinedData.EnclosedPorch = combinedData.EnclosedPorch.apply(lambda x: 1 if x else 0)

		combinedData['LotAreaMultSlope'] = combinedData.LotArea * combinedData.LandSlope

		combinedData['BsmtSFPoints'] = (combinedData.BsmtQual ** 2 + combinedData.BsmtCond + combinedData.BsmtExposure +
									combinedData.BsmtFinType1 + combinedData.BsmtFinType2)

		combinedData['BsmtSFMultPoints'] = combinedData.TotalBsmtSF * (
					combinedData.BsmtQual ** 2 + combinedData.BsmtCond + combinedData.BsmtExposure +
					combinedData.BsmtFinType1 + combinedData.BsmtFinType2)

		combinedData['TotBathrooms'] = combinedData.FullBath + (combinedData.HalfBath * 0.5) + combinedData.BsmtFullBath + (
					combinedData.BsmtHalfBath * 0.5)
		combinedData.FullBath = combinedData.FullBath.apply(lambda x: 1 if x else 0)
		combinedData.HalfBath = combinedData.HalfBath.apply(lambda x: 1 if x else 0)
		combinedData.BsmtFullBath = combinedData.BsmtFullBath.apply(lambda x: 1 if x else 0)
		combinedData.BsmtHalfBath = combinedData.BsmtHalfBath.apply(lambda x: 1 if x else 0)

		combinedData.MSSubClass = combinedData.MSSubClass.astype('str')
		combinedData.MoSold = combinedData.MoSold.astype('str')

		combinedData, dummies = self.encode(combinedData)

		zDummies = combinedData[dummies][trainShape:].sum() == 0
		combinedData.drop(dummies[zDummies], axis=1, inplace=True)
		dummies = dummies.drop(dummies[zDummies])

		zDummies = combinedData[dummies][:trainShape].sum() == 0
		combinedData.drop(dummies[zDummies], axis=1, inplace=True)
		dummies = dummies.drop(dummies[zDummies])

		del zDummies

		combinedData.YearBuilt = self.AgeInYears(combinedData.YearBuilt)
		combinedData.YearRemodAdd = self.AgeInYears(combinedData.YearRemodAdd)
		combinedData.GarageYrBlt = self.AgeInYears(combinedData.GarageYrBlt)
		combinedData.YrSold = self.AgeInYears(combinedData.YrSold)

		combinedData['Remod'] = 2
		combinedData.loc[(combinedData.YearBuilt == combinedData.YearRemodAdd), ['Remod']] = 0
		combinedData.loc[(combinedData.YearBuilt != combinedData.YearRemodAdd), ['Remod']] = 1

		combinedData["IsNew"] = 2
		combinedData.loc[(combinedData.YearBuilt == combinedData.YrSold), ['IsNew']] = 1
		combinedData.loc[(combinedData.YearBuilt != combinedData.YrSold), ['IsNew']] = 0

		combinedData.drop(
			['FireplaceQu', 'BsmtSFPoints', 'TotalBsmtSF', 'GarageArea', 'GarageCars', 'OverallQual', 'GrLivArea',
			 'TotalBsmtSF_x_Bsm', '1stFlrSF', 'PoolArea', 'LotArea', 'SaleCondition_Partial', 'Exterior1st_VinylSd',
			 'GarageCond', 'HouseStyle_2Story', 'BsmtSFMultPoints', 'ScreenPorch', 'LowQualFinSF', 'BsmtFinSF2',
			 'TSsnPorch'], axis=1, inplace=True)

		combinedData.rename(columns={'2ndFlrSF': 'SndFlrSF'}, inplace=True)

		cols = combinedData.columns
		cols = cols.drop(['SalePrice'])

		cols = cols.drop(
			['Condition1_PosN', 'Neighborhood_NWAmes', 'Exterior1st_CBlock', 'BldgType_1Fam', 'RoofStyle_Flat',
			 'MSZoning_Call', 'Alley_Grvl', 'LandContour_Bnk', 'LotConfig_Corner', 'GarageType_2Types', 'MSSubClass_45',
			 'MasVnrType_BrkCmn', 'Foundation_CBlock', 'MiscFeature_Gar2', 'SaleType_COD', 'Exterior2nd_CBlock'])

		cols = cols.drop(
			['PoolQC', 'BldgType_TwnhsE', 'BsmtFinSF1', 'BsmtUnfSF', 'Electrical_SBrkr', 'Exterior1st_MetalSd',
			 'Exterior2nd_VinylSd', 'GarageQual', 'GarageType_Attchd', 'HouseStyle_1Story', 'MasVnrType_None',
			 'MiscFeature_NA', 'MSZoning_RL', 'RoofStyle_Gable', 'SaleCondition_Normal', 'MoSold_10',
			 'SaleType_New', 'SndFlrSF', 'TotalPorchSF', 'WoodDeckSF', 'BldgType_Duplex', 'MSSubClass_90'])

		combinedData.CentralAir = combinedData.CentralAir.astype('uint8')
		combinedData.Garage_Newest = combinedData.Garage_Newest.astype('uint8')
		combinedData.EnclosedPorch = combinedData.EnclosedPorch.astype('uint8')
		combinedData.FullBath = combinedData.FullBath.astype('uint8')
		combinedData.HalfBath = combinedData.HalfBath.astype('uint8')
		combinedData.BsmtFullBath = combinedData.BsmtFullBath.astype('uint8')
		combinedData.BsmtHalfBath = combinedData.BsmtHalfBath.astype('uint8')
		combinedData.Remod = combinedData.Remod.astype('uint8')
		combinedData.IsNew = combinedData.IsNew.astype('uint8')
		combinedData.Street = combinedData.Street.astype('uint8')
		combinedData.PavedDrive = combinedData.PavedDrive.astype('uint8')
		combinedData.Functional = combinedData.Functional.astype('uint8')
		combinedData.LandSlope = combinedData.LandSlope.astype('uint8')

		numeric_features = list(combinedData.loc[:, cols].dtypes[(combinedData.dtypes != "category") & (combinedData.dtypes !='uint8')].index)

		'''
		with warnings.catch_warnings():
		    warnings.simplefilter("ignore", category=RuntimeWarning)
		'''
		skewed_features = combinedData[numeric_features].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)

		skewness = pd.DataFrame({'Skew' :skewed_features})

		skewness = skewness[abs(skewness) > 0.7]
		skewness = skewness.dropna()

		l_opt = {}

		for feat in skewness.index:
		    combinedData[feat], l_opt[feat] = boxcox((combinedData[feat]+1))

		poly_cols = ['ConstructArea', 'TotalPoints', 'LotAreaMultSlope', 'GarageArea_x_Car']

		pf = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
		res = pf.fit_transform(combinedData.loc[:, poly_cols])

		target_feature_names = [feat.replace(' ', '_') for feat in pf.get_feature_names(poly_cols)]
		output_df = pd.DataFrame(res, columns=target_feature_names, index=combinedData.index).iloc[:, len(poly_cols):]
		print('Polynomial Features included:', output_df.shape[1])
		combinedData = pd.concat([combinedData, output_df], axis=1)
		print('Total Features after Polynomial Features included:', combinedData.shape[1])
		colsP = output_df.columns

		del output_df, target_feature_names, res, pf

		y_train = (combinedData.SalePrice[combinedData.SalePrice>0].reset_index(drop=True, inplace=False))

		return combinedData, y_train, cols, colsP

	def encode(self, df):
		categorical_cols = df.select_dtypes(include=['object']).columns

		print(len(categorical_cols), "categorical columns")
		print(categorical_cols)
		for col in categorical_cols:
			df[col] = df[col].str.replace('\W', '').str.replace(' ', '_')

		dummies = pd.get_dummies(df[categorical_cols], columns=categorical_cols).columns
		df = pd.get_dummies(df, columns=categorical_cols)

		print("Total Columns:", len(df.columns))
		print(df.info())

		return df, dummies

	def AgeInYears(self, feature):
		return feature.apply(lambda x: 0 if x == 0 else (2011 - x))
