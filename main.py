# data analysis libraries
import numpy as np
import pandas as pd
import scipy.stats

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# import data using .read_csv()
train = pd.read_csv("Datasets/merged.csv")
treshank = train[train['owner'] == 'Treshank']
tushar = train[train['owner'] == 'Tushar']
vaibhav = train[train['owner'] == 'Vaibhav']

train_numeric = train.drop(['id','name','artists'], axis=1)

treshank_numeric = treshank.drop(['id','name','artists'], axis=1)
treshank_small = treshank_numeric.drop(['tempo','duration_ms','key','loudness','time_signature'], axis=1)

tushar_numeric = tushar.drop(['id','name','artists'], axis=1)
tushar_small = tushar_numeric.drop(['tempo','duration_ms','key','loudness','time_signature'], axis=1)

vaibhav_numeric = vaibhav.drop(['id','name','artists'], axis=1)
vaibhav_small = vaibhav_numeric.drop(['tempo','duration_ms','key','loudness','time_signature'], axis=1)


treshank_means = pd.DataFrame(treshank_small.mean(axis=0)).T
tushar_means = pd.DataFrame(tushar_small.mean(axis=0)).T
vaibhav_means = pd.DataFrame(vaibhav_small.mean(axis=0)).T
treshank_means['owner'] = 'Treshank'
tushar_means['owner'] = 'Tushar'
vaibhav_means['owner'] = 'Vaibhav'
means = vaibhav_means.append(tushar_means.append(treshank_means))

# create a palette from hex codes and set palette
bright = ["#F8766D", "#00BFC4", "#FFC400", "#03ED3A", "#003FFF", "#8A2BE2"]
sns.set_palette(bright)

new = means.melt('owner', var_name='cols',  value_name='vals')
sns.factorplot(x="cols", y="vals", hue='owner', data=new, kind='bar', size=6, legend_out=False)
plt.xticks(rotation=90, fontsize=12) # set audio feature labels
plt.ylabel("Value", fontsize=15) # set y axis label
plt.xlabel("Audio Features", fontsize=15) # set x axis label
plt.title("Mean Audio Features by Owner", fontsize = 17) # set chart title
plt.legend(fontsize=12) # increase legend fontsize
plt.show() # remove text output

# set figure size
sns.set(rc={'figure.figsize':(20,13)})

# Danceability
plt.subplot(421)
sns.distplot(treshank['danceability'], label='Treshank')
sns.distplot(tushar['danceability'], color='r', label='Tushar')
sns.distplot(vaibhav['danceability'], color='g', label='Vaibhav')
plt.xlabel('DANCEABILITY', fontsize=12)
plt.legend(fontsize=12)

# Energy
plt.subplot(422)
sns.distplot(treshank['energy'], label='Treshank')
sns.distplot(tushar['energy'], color='r', label='Tushar')
sns.distplot(vaibhav['energy'], color='g', label='Vaibhav')
plt.xlabel('ENERGY', fontsize=13)
plt.legend(fontsize=13)

# Mode
plt.subplot(423)
sns.distplot(treshank['mode'], label='Treshank')
sns.distplot(tushar['mode'], color='r', label='Tushar')
sns.distplot(vaibhav['mode'], color='g', label='Vaibhav')
plt.xlabel('MODE', fontsize=13)
plt.legend(fontsize=13)

# Speechiness
plt.subplot(424)
sns.distplot(treshank['speechiness'], label='Treshank')
sns.distplot(tushar['speechiness'], color='r', label='Tushar')
sns.distplot(vaibhav['speechiness'], color='g', label='Vaibhav')
plt.xlabel('SPEECHINESS', fontsize=13)
plt.legend(fontsize=13)

# Acousticness
plt.subplot(425)
sns.distplot(treshank['acousticness'], label='Treshank')
sns.distplot(tushar['acousticness'], color='r', label='Tushar')
sns.distplot(vaibhav['acousticness'], color='g', label='Vaibhav')
plt.xlabel('ACOUSTICNESS', fontsize=13)
plt.legend(fontsize=13)

# Instrumentalness
plt.subplot(426)
sns.distplot(treshank['instrumentalness'], label='Treshank').set(ylim=(0, 10))
sns.distplot(tushar['instrumentalness'], color='r', label='Tushar').set(ylim=(0, 10))
sns.distplot(vaibhav['instrumentalness'], color='g', label='Vaibhav').set(ylim=(0, 10))
plt.xlabel('INSTRUMENTALNESS', fontsize=13)
plt.legend(fontsize=13)

# Liveness
plt.subplot(427)
sns.distplot(treshank['liveness'], label='treshank')
sns.distplot(tushar['liveness'], color='r', label='Tushar')
sns.distplot(vaibhav['liveness'], color='g', label='Vaibhav')
plt.xlabel('LIVENESS', fontsize=13)
plt.legend(fontsize=13)

# Valence
plt.subplot(428)
sns.distplot(treshank['valence'], label='treshank')
sns.distplot(tushar['valence'], color='r', label='Tushar')
sns.distplot(vaibhav['valence'], color='g', label='Vaibhav')
plt.xlabel('VALENCE', fontsize=13)
plt.legend(fontsize=13)

plt.tight_layout()
plt.show()

plt.figure(figsize = (16,5))
sns.heatmap(train_numeric.corr(), cmap="coolwarm", annot=True)
plt.show()


features = train.drop(['owner','id','name','artists'], axis=1)
target = train['owner']

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size= 0.15)
print('Features Training Set:', x_train.shape, 'Features Testing Set:', x_test.shape)
print('Target Training Set:', y_train.shape, 'Target Testing Set:', y_test.shape)

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
acc_randomforest = accuracy_score(y_pred, y_test) * 100
print("Random Forest:", acc_randomforest)

# Grid Search to find Best Parameters
param_grid = {
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_cv = GridSearchCV(estimator=randomforest, param_grid=param_grid)
rf_cv.fit(x_train, y_train)

print(rf_cv.best_params_)

y_pred = rf_cv.predict(x_test)
acc_rfcv = accuracy_score(y_pred, y_test)*100
print("Random Forest GridSearchCV:", acc_rfcv)


testdf = pd.read_csv("Datasets/testdf.csv")
owners = ['Tushar', 'Tushar', 'Tushar', 'Tushar', 'Tushar', 'Treshank', 'Treshank', 'Treshank', 'Treshank']
prediction = rf_cv.predict(testdf)
accuracy = accuracy_score(prediction, owners)
print(prediction)
print("Correct Predictions: %d/%d" % (accuracy*len(owners), len(owners)))
print("Accuracy: %.2f%%" % (accuracy*100))

