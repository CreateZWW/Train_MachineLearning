import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from scipy.stats.stats import pearsonr
import pandas as pd
import sys

pd.set_option('display.max_rows',10)
pd.set_option('display.max_columns',20)
pd.set_option('display.width',1000)

# 读取数据
def LoadData():
	p_dataTrain = pd.read_csv('Datasets//train.csv', index_col=0)
	p_dataTest = pd.read_csv('Datasets//test.csv', index_col=0)
	p_dataAll = pd.concat(
		[p_dataTrain.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], p_dataTest.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]])
	return p_dataTrain, p_dataTest, p_dataAll


# 数据统计
def StatisticData(p_data):
	# 统计各列缺失值
	count = len(p_data)
	print("////////////////////////////////////")
	print("All " + str(count))
	print("NULL value statistic result")
	for col in p_data.columns:
		temNum = p_data[col].isnull().sum()
		print(str(col) + " " + str(temNum))


# 数据清理
def PreprocessingData(p_dataTrain, p_dataTest, p_dataAll):
	p_dataAll = pd.DataFrame(p_dataAll)
	for index, row in p_dataAll.iterrows():
		df = pd.DataFrame(row)
		# 对于cabin，空缺填0，无空缺填1
		if str(df.iloc[8, 0]) == 'nan' or str(df.iloc[8, 0]) == 'NaN':
			p_dataAll.loc[index, 'Cabin'] = 0
			if index < 892:
				p_dataTrain.loc[index, 'Cabin'] = 0
			else:
				p_dataTest.loc[index, 'Cabin'] = 0
		else:
			p_dataAll.loc[index, 'Cabin'] = 1
			if index < 892:
				p_dataTrain.loc[index, 'Cabin'] = 1
			else:
				p_dataTest.loc[index, 'Cabin'] = 1
		# 对于Embarked和Fare，如果是train丢掉有空缺值的;如果是test,Fare取平均，Embarked取众数
		if str(df.iloc[7, 0]) == 'nan' or str(df.iloc[7, 0]) == 'NaN':
			p_dataAll = p_dataAll.drop(index, axis=0)
			if index < 892:
				p_dataTrain = p_dataTrain.drop(index, axis=0)
			else:
				p_dataTest.loc[index, 'Fare'] = p_dataTest['Fare'].mean()
			continue
		if str(df.iloc[9, 0]) == 'nan' or str(df.iloc[9, 0]) == 'NaN':
			p_dataAll = p_dataAll.drop(index, axis=0)
			if index < 892:
				p_dataTrain = p_dataTrain.drop(index, axis=0)
			else:
				p_dataTest.loc[index, 'Embarked'] = p_dataTest['Embarked'].mode()[0]
			continue
	#随即森林算法填补年龄缺失
	age_df = p_dataAll.loc[:, ['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
	# 乘客分成已知年龄和未知年龄两部分
	known_age = age_df[age_df.Age.notnull()].values
	unknown_age = age_df[age_df.Age.isnull()].values
	# y即目标年龄
	y = known_age[:, 0]
	# X即特征属性值
	X = known_age[:, 1:]
	# fit到RandomForestRegressor之中
	rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
	rfr.fit(X, y)
	# 用得到的模型进行未知年龄结果预测
	predictedAges = rfr.predict(unknown_age[:, 1::])
	# 用得到的预测结果填补原缺失数据
	p_dataAll.loc[(p_dataAll.Age.isnull()), 'Age'] = predictedAges
	#补充的年龄填回去
	for index, row in p_dataAll.iterrows():
		if index < 892:
			p_dataTrain.loc[index, 'Age'] = p_dataAll.loc[index, 'Age']
		else:
			p_dataTest.loc[index, 'Age'] = p_dataAll.loc[index, 'Age']
	return p_dataTrain, p_dataTest, p_dataAll


def SVMPredict(p_dataTrain, p_dataTest):
	#embark和sex用数字来替代
	le = sklearn.preprocessing.LabelEncoder()
	le.fit(pd.unique(p_dataTrain.Sex))
	sex_t = le.transform(p_dataTrain.Sex)
	test_sex_t = le.transform(p_dataTest.Sex)
	le = sklearn.preprocessing.LabelEncoder()
	le.fit(pd.unique(p_dataTrain.Embarked))
	embarked_t = le.transform(p_dataTrain.Embarked)
	test_embarked_t = le.transform(p_dataTest.Embarked)
	#提取Survived然后造新df
	Y = p_dataTrain.iloc[:, 0]
	train = p_dataTrain.iloc[:, [1, 3, 4, 5, 6, 8, 9, 10]]
	test = p_dataTest.iloc[:, [0, 2, 3, 4, 5, 7, 8, 9]]
	train.loc[:, 'Sex'] = sex_t
	test.loc[:, 'Sex'] = test_sex_t
	train.loc[:, 'Embarked'] = embarked_t
	test.loc[:, 'Embarked'] = test_embarked_t
	#数据归一化
	scaler = StandardScaler()
	scaler.fit(train.values)
	X_train = scaler.transform(train)
	X_test = scaler.transform(test)
	#svm计算
	svm = SVC()
	parameters = {'kernel': ('linear', 'rbf'), 'C': (1, 0.25, 0.5, 0.75), 'gamma': (1, 2, 3, 'auto'),
				  'decision_function_shape': ('ovo', 'ovr'), 'shrinking': (True, False)}
	clf = GridSearchCV(svm, parameters)
	clf.fit(X_train, Y)
	print("accuracy:" + str(np.average(cross_val_score(clf, X_train, Y, scoring='accuracy'))))
	print("f1:" + str(np.average(cross_val_score(clf, X_train, Y, scoring='f1'))))
	result = clf.predict(X_test)
	result = pd.DataFrame(result)
	result.to_csv("result.csv")
	#clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
	#clf.fit()


[dataTrain, dataTest, dataAll] = LoadData()
[dataTrain, dataTest, dataAll] = PreprocessingData(dataTrain, dataTest, dataAll)
SVMPredict(dataTrain, dataTest)