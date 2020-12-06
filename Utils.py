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


class Utils:
    @staticmethod
    def printToCSV(df, fileName):
        path = 'D:\\SourceFiles\\MachineLearning-Fall2020\\HousePrices\\' + fileName
        df.to_csv(index=False, path_or_buf=path)

    @staticmethod
    def printDatasetInfo(dataset, label=''):
        if (label == ''):
            dataset.info()
        else:
            print(dataset[label])

    @staticmethod
    def plotData(df, xClass, yClass):
        fig = plt.figure(figsize=(20, 15))
        sns.set(font_scale=1.5)

        fig1 = fig.add_subplot(221);
        sns.boxplot(x=xClass, y=yClass, data=df[[yClass, xClass]])

        fig2 = fig.add_subplot(222);
        sns.scatterplot(x=df[xClass], y=df[yClass], hue=df[xClass], palette='Spectral')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def printDatasetNulls(dataset):
        print("Data nulls:", dataset.isnull().sum().sum())

    @staticmethod
    def fillNullLabels(dataset, labels, fillValue):
        for label in labels:
            dataset[label] = dataset[label].fillna(fillValue)

        return dataset



