###########################################################
# imports
import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

###########################################################
# options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

###########################################################
# datasets
price = pd.read_csv("datasets/price.csv")
train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")


###########################################################
# functions
def outlier_thresholds(dataframe, variable, qmin, qmax):
    quartile1 = dataframe[variable].quantile(qmin)
    quartile3 = dataframe[variable].quantile(qmax)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + (1.5 * interquantile_range)
    low_limit = quartile1 - (1.5 * interquantile_range)
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable, qmin=0.25, qmax=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, qmin, qmax)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


###########################################################
# brent petrol
brent = pd.read_csv("datasets/brent_petrol.csv")
brent.columns = ['date', 'now', 'open', 'high', 'low', 'volume', 'diff']
brent["date"] = pd.to_datetime(brent["date"], dayfirst=True)
brent["now"] = brent["now"].str.replace(",", ".")
brent["now"] = brent["now"].astype(float)
brent = brent[["date", "now"]]
###########################################################
# dollar
dolar = pd.read_csv("datasets/dolar.csv")
dolar.columns = ['date', 'now', 'open', 'high', 'low', 'volume', 'diff']
dolar["date"] = pd.to_datetime(dolar["date"], dayfirst=True)
dolar["now"] = dolar["now"].str.replace(",", ".")
dolar["now"] = dolar["now"].astype(float)
dolar = dolar[["date", "now"]]


###########################################################
# data preprocessing
price["Validfrom"] = pd.to_datetime(price["Validfrom"])
train["Billing Date"] = pd.to_datetime(train["Billing Date"])
test["Week Start Date"] = pd.to_datetime(test["Week Start Date"])

pr = price.groupby(["Validfrom", "City Code", "Product"]).agg({"CRP": "mean"})
pr = pr.reset_index()

brent.columns = ['Billing Date', 'now_brent']
dolar.columns = ['Billing Date', 'now_dolar']
pr.columns = ["Billing Date", "City Code", "Product", "CRP"]
train = train.merge(pr, how="left")
train = train.merge(brent, how="left")
train = train.merge(dolar, how="left")
brent.columns = ["Week Start Date", 'now_brent']
dolar.columns = ["Week Start Date", 'now_dolar']
pr.columns = ["Week Start Date", "City Code", "Product", "CRP"]
test = test.merge(pr, how="left")
test = test.merge(brent, how="left")
test = test.merge(dolar, how="left")
train["CRP"].fillna(train["CRP"].mean(), inplace=True)
test["CRP"].fillna(test["CRP"].mean(), inplace=True)
train.loc[:, "now_brent"] = train.loc[:, "now_brent"].fillna(train["now_brent"].mean())
train.loc[:, "now_dolar"] = train.loc[:, "now_dolar"].fillna(train["now_dolar"].mean())
test.loc[:, "now_brent"] = test.loc[:, "now_brent"].fillna(test["now_brent"].mean())
test.loc[:, "now_dolar"] = test.loc[:, "now_dolar"].fillna(test["now_dolar"].mean())
train.isnull().sum()
test.isnull().sum()
train.describe().T
train = train[train["Litres"] > 0]
replace_with_thresholds(train, "Litres", 0.01, 0.99)
train.describe().T

train["week"] = train["Billing Date"].dt.week
train["year"] = train["Billing Date"].dt.year
train.drop(["Billing Date", "Plant"], axis=1, inplace=True)
train

test["week"] = test["Week Start Date"].dt.week
test["year"] = test["Week Start Date"].dt.year
test.drop(["index", "Week Start Date"], axis=1, inplace=True)
test

###########################################################
# vehicle counts
#############################################

train.loc[train["year"] == 2014, "vehicle_counts"] = 18828721
train.loc[(train["year"] == 2014) & (train["Product"] == "Benzin"), "arac_yakit_tipi"] = 2855078
train.loc[(train["year"] == 2014) & (train["Product"] == "Motorin"), "arac_yakit_tipi"] = 2882885

train.loc[train["year"] == 2015, "vehicle_counts"] = 19994472
train.loc[(train["year"] == 2015) & (train["Product"] == "Benzin"), "arac_yakit_tipi"] = 2927720
train.loc[(train["year"] == 2015) & (train["Product"] == "Motorin"), "arac_yakit_tipi"] = 3345951

train.loc[train["year"] == 2016, "vehicle_counts"] = 21090424
train.loc[(train["year"] == 2016) & (train["Product"] == "Benzin"), "arac_yakit_tipi"] = 3031744
train.loc[(train["year"] == 2016) & (train["Product"] == "Motorin"), "arac_yakit_tipi"] = 3803772

train.loc[train["year"] == 2017, "vehicle_counts"] = 22218945
train.loc[(train["year"] == 2017) & (train["Product"] == "Benzin"), "arac_yakit_tipi"] = 3120407
train.loc[(train["year"] == 2017) & (train["Product"] == "Motorin"), "arac_yakit_tipi"] = 4256305

train.loc[train["year"] == 2018, "vehicle_counts"] = 22865921
train.loc[(train["year"] == 2018) & (train["Product"] == "Benzin"), "arac_yakit_tipi"] = 3089626
train.loc[(train["year"] == 2018) & (train["Product"] == "Motorin"), "arac_yakit_tipi"] = 4568665


test.loc[test["year"] == 2019, "vehicle_counts"] = 23156975
test.loc[(test["year"] == 2019) & (test["Product"] == "Benzin"), "arac_yakit_tipi"] = 3020017
test.loc[(test["year"] == 2019) & (test["Product"] == "Motorin"), "arac_yakit_tipi"] = 4769714


###########################################################
# model LinearRegression()
train = train[train["City Code"].isin(test["City Code"])]
train["City Code"] = train["City Code"].astype(str)
test["City Code"] = test["City Code"].astype(str)

# tarihler
train["mevsim"] = "x"
train.loc[((train["week"] < 13) | (train["week"] > 39)), "mevsim"] = "kis"
train.loc[((train["week"] >= 13) & (train["week"] <= 39)), "mevsim"] = "yaz"

test.loc[((test["week"] < 13) | (test["week"] > 39)), "mevsim"] = "kis"
test.loc[((test["week"] >= 13) & (test["week"] <= 39)), "mevsim"] = "yaz"

############################
# ramazan
# kurban
train.loc[(train["year"] == 2014) & ((train["week"] == 30) | (train["week"] == 31)), "is_bayram"] = 1
train.loc[(train["year"] == 2014) & ((train["week"] == 40) | (train["week"] == 41)), "is_bayram"] = 1

train.loc[(train["year"] == 2015) & ((train["week"] == 29) | (train["week"] == 30)), "is_bayram"] = 1
train.loc[(train["year"] == 2015) & ((train["week"] == 39) | (train["week"] == 40)), "is_bayram"] = 1

train.loc[(train["year"] == 2016) & ((train["week"] == 26) | (train["week"] == 27)), "is_bayram"] = 1
train.loc[(train["year"] == 2016) & ((train["week"] == 36) | (train["week"] == 37)), "is_bayram"] = 1

train.loc[(train["year"] == 2017) & ((train["week"] == 25) | (train["week"] == 26)), "is_bayram"] = 1
train.loc[(train["year"] == 2017) & ((train["week"] == 35) | (train["week"] == 36)), "is_bayram"] = 1

train.loc[(train["year"] == 2018) & ((train["week"] == 24) | (train["week"] == 25)), "is_bayram"] = 1
train.loc[(train["year"] == 2018) & ((train["week"] == 33) | (train["week"] == 34)), "is_bayram"] = 1

test.loc[(test["year"] == 2019) & ((test["week"] == 22) | (test["week"] == 23)), "is_bayram"] = 1
test.loc[(test["year"] == 2019) & ((test["week"] == 32) | (test["week"] == 33)), "is_bayram"] = 1

train.isnull().sum()
test.isnull().sum()
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)



dm_train = pd.get_dummies(train)
dm_test = pd.get_dummies(test)

dm_train.columns
dm_train.isnull().sum().sum()
dm_test.isnull().sum().sum()
reg_model = LinearRegression()

X = dm_train.drop("Litres", axis=1)
y = dm_train[["Litres"]]
reg_model.fit(X, y)

train[~train["City Code"].isin(test["City Code"])]

predictions = test.copy()
predictions["Litres"] = reg_model.predict(dm_test)

submissions = predictions[["Litres"]]
submissions.to_csv("heycanli.csv", index_label="index")

train.corr()
train.loc[train["mevsim"] == "kis", "mevsim"] = 0
train.loc[train["mevsim"] == "yaz", "mevsim"] = 1

train["mevsim"] = train["mevsim"].astype(int)
train.corr()

reg_model.coef_ = x
x =reg_model.coef_
x[0][0] = 0
dm_train.columns