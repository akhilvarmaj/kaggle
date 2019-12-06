import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('C:/Users/uib43221/.spyder-py3/Kaggle/house-prices-advanced-regression-techniques/train.csv')
data_test = pd.read_csv('C:/Users/uib43221/.spyder-py3/Kaggle/house-prices-advanced-regression-techniques/test_data.csv')
data.head(5)

[[data.shape]]

data.info

type(data)
data.dtypes
d = pd.DataFrame(data.dtypes)

plt.figure(figsize=(15,5))
sns.heatmap(data.isnull(),yticklabels=False,cmap='winter')


sns.heatmap(data_test.isnull(),yticklabels=False,cmap='gist_rainbow_r')


data = data.drop(['Alley','PoolQC','Fence', 'MiscFeature'],axis=1)
data = data.drop(['Id'],axis=1)
data.head()

data.columns
data.isnull().any().any()


data['FireplaceQu'].isnull().sum()
data['Electrical'].dtype

data['FireplaceQu'] = data['FireplaceQu'].fillna(data['FireplaceQu'].mode()[0])

d = pd.DataFrame(data.isnull().sum())

data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mode()[0])

data['GarageType'] = data['GarageType'].fillna(data['GarageType'].mode()[0])
data['GarageFinish'] = data['GarageFinish'].fillna(data['GarageFinish'].mode()[0])
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].mean())
data['GarageQual'] = data['GarageQual'].fillna(data['GarageQual'].mode()[0])
data['GarageCond'] = data['GarageCond'].fillna(data['GarageCond'].mode()[0])
data['BsmtCond'] = data['BsmtCond'].fillna(data['BsmtCond'].mode()[0])
data['BsmtExposure'] = data['BsmtExposure'].fillna(data['BsmtExposure'].mode()[0])
data['BsmtFinType1'] = data['BsmtFinType1'].fillna(data['BsmtFinType1'].mode()[0])
data['BsmtFinType2'] = data['BsmtFinType2'].fillna(data['BsmtFinType2'].mode()[0])
data['BsmtQual'] = data['BsmtQual'].fillna(data['BsmtQual'].mode()[0])
data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].mean())
data['MasVnrType'] = data['MasVnrType'].fillna(data['MasVnrType'].mode()[0])
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mode()[0])

y = data['SalePrice']

data = data.drop(['SalePrice'],axis=1)

global_data = pd.concat([data,data_test],axis=0)
global_data.shape
global_data.isnull().any().any()
sns.heatmap(global_data.isnull(),yticklabels=False,cmap='viridis')
d1 = pd.DataFrame(global_data.isnull().sum())
global_data['Electrical'] = global_data['Electrical'].fillna(global_data['Electrical'].mode()[0])


columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

def category_onehot_multcols(multcolumns):
    global_dataf = global_data
    i=0
    for fields in multcolumns:
        
        print(fields)
        global_data1=pd.get_dummies(global_data[fields],drop_first=True)
        #global_data.drop([fields],axis=1)
        
        if i==0:
            global_dataf=global_data1.copy()
            
        else:
            global_dataf = pd.concat([global_dataf,global_data1],axis=1)
            
        i=i+1
        
    global_dataf = pd.concat([global_data,global_dataf],axis=1)
    
    return global_dataf


main_df = global_data.copy()

global_dataf=category_onehot_multcols(columns)

global_dataf.shape

global_dataf =global_dataf.loc[:,~global_dataf.columns.duplicated()]

global_dataf = global_dataf.drop(['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive'],axis=1)


global_dataf.dtyes

df_Train=global_dataf.iloc[:1422,:]
df_Test=global_dataf.iloc[1422:,:]


type(df_Test)
type(y)

X_train=df_Train
X_test = df_Test
y_train = pd.DataFrame(y)

from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor(learning_rate=0.1,n_estimators=100,subsample=1.0,)
clf.fit(X_train,y_train)
preds = clf.predict(df_Test)
preds = pd.DataFrame(preds)














#import xgboost
#classifier=xgboost.XGBRegressor()

#booster=['gbtree','gblinear']
#base_score=[0.25,0.5,0.75,1]


## Hyper Parameter Optimization


#n_estimators = [100, 500, 900, 1100, 1500]
#max_depth = [2, 3, 5, 10, 15]
#booster=['gbtree','gblinear']
#learning_rate=[0.05,0.1,0.15,0.20]
#min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
#hyperparameter_grid = {
#   'n_estimators': n_estimators,
#  'max_depth':max_depth,
#    'learning_rate':learning_rate,
#    'min_child_weight':min_child_weight,
#    'booster':booster,
#   'base_score':base_score
#    }


# Set up the random search with 4-fold cross validation
#random_cv = RandomizedSearchCV(estimator=regressor,
#            param_distributions=hyperparameter_grid,
#            cv=5, n_iter=50,
#            scoring = 'neg_mean_absolute_error',n_jobs = 4,
#            verbose = 5, 
#            return_train_score = True,
#            random_state=42)


#random_cv.fit(X_train,y_train)
#random_cv.best_estimator_
#regressor.fit(X_train,y_train)