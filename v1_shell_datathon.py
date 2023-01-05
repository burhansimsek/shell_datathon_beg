###############################################################################
# Will the BG/NBD model-Gamma Gamma Submodel work?
###############################################################################
# imports
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.10)
    quartile3 = dataframe[variable].quantile(0.90)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


########################################
# Test Dataset
test = pd.read_csv("../datasets/test.csv")
test.head()
test.columns
test["Week Start Date"].nunique()
test["Week Start Date"].min()
test["Week Start Date"].max()

test_motorin = test[test["Product"] == "Motorin"]
test_benzin = test[test["Product"] == "Benzin"]

########################################
# First Look at Data
df = pd.read_csv("../datasets/train.csv")
df.head(10)
df.info()
df.describe().T
df.isnull().sum()

df_motorin = df[df["Product"] == "Motorin"]
df_benzin = df[df["Product"] == "Benzin"]
########################################
# Data Preprocessing
replace_with_thresholds(df_motorin, "Litres")
replace_with_thresholds(df_benzin, "Litres")

df_motorin = df_motorin[df_motorin["Litres"] > 0]
df_benzin = df_benzin[df_benzin["Litres"] > 0]

df_motorin['Billing Date'] = pd.to_datetime(df_motorin['Billing Date'])
df_benzin['Billing Date'] = pd.to_datetime(df_benzin['Billing Date'])

########################################################################################################################
# CLTV
# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç
# ['Customer', 'Area', 'City Code', 'Billing Date', 'Product', 'Plant', 'Litres']
########################################################################################################################
# Motorin
today_date = df_motorin["Billing Date"].max()

cltv_motorin = df_motorin.groupby("Customer").agg({"Billing Date": [lambda x: (x.max() - x.min()).days,
                                                                    lambda x: (today_date - x.min()).days],
                                                   "Plant": lambda x: x.count(),
                                                   "Litres": lambda x: x.sum()
                                                   })

cltv_motorin.columns = ["recency", "T", "frequency", "monetary"]

cltv_motorin["monetary"] = cltv_motorin["monetary"] / cltv_motorin["frequency"]
cltv_motorin["recency"] = cltv_motorin["recency"] / 7
cltv_motorin["T"] = cltv_motorin["T"] / 7
cltv_motorin
cltv_motorin = cltv_motorin[cltv_motorin["recency"] > 0]
cltv_motorin.describe().T
########################################################################################################################
# Benzin
today_date = df_benzin["Billing Date"].max()

cltv_benzin = df_benzin.groupby("Customer").agg({"Billing Date": [lambda x: (x.max() - x.min()).days,
                                                                  lambda x: (today_date - x.min()).days],
                                                 "Plant": lambda x: x.count(),
                                                 "Litres": lambda x: x.sum()
                                                 })

cltv_benzin.columns = ["recency", "T", "frequency", "monetary"]

cltv_benzin["monetary"] = cltv_benzin["monetary"] / cltv_benzin["frequency"]
cltv_benzin["recency"] = cltv_benzin["recency"] / 7
cltv_benzin["T"] = cltv_benzin["T"] / 7
cltv_benzin
cltv_benzin = cltv_benzin[cltv_benzin["recency"] > 0]
cltv_benzin.describe().T
cltv_motorin.describe().T

########################################
# BG/NBD Model Motorin

bgf_motorin = BetaGeoFitter(penalizer_coef=0.02)

bgf_motorin.fit(cltv_motorin['frequency'],
                cltv_motorin['recency'],
                cltv_motorin['T'])

a_motorin = pd.DataFrame()
a_motorin["1_week_transaction"] = bgf_motorin.predict(1,
                                                      cltv_motorin['frequency'],
                                                      cltv_motorin['recency'],
                                                      cltv_motorin['T'])
########################################
# BG/NBD Model Benzin

bgf_benzin = BetaGeoFitter(penalizer_coef=0.02)

bgf_benzin.fit(cltv_benzin['frequency'],
               cltv_benzin['recency'],
               cltv_benzin['T'])

a_benzin = pd.DataFrame()
a_benzin["1_week_transaction"] = bgf_benzin.predict(1,
                                                    cltv_benzin['frequency'],
                                                    cltv_benzin['recency'],
                                                    cltv_benzin['T'])

########################################
# Gamma Gamma Model Motorin

ggf_motorin = GammaGammaFitter(penalizer_coef=0.02)

ggf_motorin.fit(cltv_motorin['frequency'], cltv_motorin['monetary'])

a_motorin["expected_average_profit"] = ggf_motorin.conditional_expected_average_profit(cltv_motorin['frequency'],
                                                                                       cltv_motorin['monetary'])

a_motorin.head()
########################################
# Gamma Gamma Model Benzin

ggf_benzin = GammaGammaFitter(penalizer_coef=0.02)

ggf_benzin.fit(cltv_benzin['frequency'], cltv_benzin['monetary'])

a_benzin["expected_average_profit"] = ggf_benzin.conditional_expected_average_profit(cltv_benzin['frequency'],
                                                                                     cltv_benzin['monetary'])

a_benzin.head()

#######################################
# Motorin
for i in range(1, 14):
    a_motorin[str(i) + "_month"] = ggf_motorin.customer_lifetime_value(bgf_motorin,
                                                                       cltv_motorin['frequency'],
                                                                       cltv_motorin['recency'],
                                                                       cltv_motorin['T'],
                                                                       cltv_motorin['monetary'],
                                                                       time=i,  # aylık
                                                                       freq="W"  # T'nin frekans bilgisi.
                                                                       )

a_motorin.head()
month = [col for col in a_motorin.columns if "mon" in col]
a_motorin[month] = a_motorin[month] / 4
a_motorin.head()

a_motorin.drop(["1_week_transaction", "expected_average_profit"], axis=1, inplace=True)
i = 12
a_motorin.iloc[:, 1]
while i != 0:
    a_motorin.iloc[:, i] = a_motorin.iloc[:, i] - a_motorin.iloc[:, i - 1]
    i = i - 1

#######################################
# Benzin
for i in range(1, 14):
    a_benzin[str(i) + "_month"] = ggf_benzin.customer_lifetime_value(bgf_benzin,
                                                                     cltv_benzin['frequency'],
                                                                     cltv_benzin['recency'],
                                                                     cltv_benzin['T'],
                                                                     cltv_benzin['monetary'],
                                                                     time=i,  # aylık
                                                                     freq="W"  # T'nin frekans bilgisi.
                                                                     )

a_benzin.head()
month = [col for col in a_benzin.columns if "mon" in col]
a_benzin[month] = a_benzin[month] / 4
a_benzin.head()

a_benzin.drop(["1_week_transaction", "expected_average_profit"], axis=1, inplace=True)
i = 12
a_benzin.iloc[:, 1]
while i != 0:
    a_benzin.iloc[:, i] = a_benzin.iloc[:, i] - a_benzin.iloc[:, i - 1]
    i = i - 1

#######################################################################################################################
# Motorin

trying = pd.DataFrame({k: [np.arange(1, 53)] for k in a_motorin.groupby("Customer").agg("sum").index})
trying = trying.T
trying.columns = ["week"]
trying.head()
trying = trying.explode("week")
trying = trying.reset_index()
trying.columns = ["Customer", "Week"]
trying
trying["Week"] = trying["Week"].astype(int)

new_motorin = pd.merge(trying, a_motorin.reset_index(), how="left")

new_motorin.loc[new_motorin["Week"].isin([1, 2, 3, 4]), "predict"] = new_motorin["1_month"]
new_motorin.loc[new_motorin["Week"].isin([5, 6, 7, 8]), "predict"] = new_motorin["2_month"]
new_motorin.loc[new_motorin["Week"].isin([9, 10, 11, 12]), "predict"] = new_motorin["3_month"]
new_motorin.loc[new_motorin["Week"].isin([13, 14, 15, 16]), "predict"] = new_motorin["4_month"]
new_motorin.loc[new_motorin["Week"].isin([17, 18, 19, 20]), "predict"] = new_motorin["5_month"]
new_motorin.loc[new_motorin["Week"].isin([21, 22, 23, 24]), "predict"] = new_motorin["6_month"]
new_motorin.loc[new_motorin["Week"].isin([25, 26, 27, 28]), "predict"] = new_motorin["7_month"]
new_motorin.loc[new_motorin["Week"].isin([29, 30, 31, 32]), "predict"] = new_motorin["8_month"]
new_motorin.loc[new_motorin["Week"].isin([33, 34, 35, 36]), "predict"] = new_motorin["9_month"]
new_motorin.loc[new_motorin["Week"].isin([37, 38, 39, 40]), "predict"] = new_motorin["10_month"]
new_motorin.loc[new_motorin["Week"].isin([41, 42, 43, 44]), "predict"] = new_motorin["11_month"]
new_motorin.loc[new_motorin["Week"].isin([45, 46, 47, 48]), "predict"] = new_motorin["12_month"]
new_motorin.loc[new_motorin["Week"].isin([49, 50, 51, 52]), "predict"] = new_motorin["13_month"]

last_motorin = new_motorin[["Customer", "predict"]]
last_motorin
#######################################################################################################################
# Benzin

trying = pd.DataFrame({k: [np.arange(1, 53)] for k in a_benzin.groupby("Customer").agg("sum").index})
trying = trying.T
trying.columns = ["week"]
trying.head()
trying = trying.explode("week")
trying = trying.reset_index()
trying.columns = ["Customer", "Week"]
trying
trying["Week"] = trying["Week"].astype(int)

new_benzin = pd.merge(trying, a_benzin.reset_index(), how="left")

new_benzin.loc[new_benzin["Week"].isin([1, 2, 3, 4]), "predict"] = new_benzin["1_month"]
new_benzin.loc[new_benzin["Week"].isin([5, 6, 7, 8]), "predict"] = new_benzin["2_month"]
new_benzin.loc[new_benzin["Week"].isin([9, 10, 11, 12]), "predict"] = new_benzin["3_month"]
new_benzin.loc[new_benzin["Week"].isin([13, 14, 15, 16]), "predict"] = new_benzin["4_month"]
new_benzin.loc[new_benzin["Week"].isin([17, 18, 19, 20]), "predict"] = new_benzin["5_month"]
new_benzin.loc[new_benzin["Week"].isin([21, 22, 23, 24]), "predict"] = new_benzin["6_month"]
new_benzin.loc[new_benzin["Week"].isin([25, 26, 27, 28]), "predict"] = new_benzin["7_month"]
new_benzin.loc[new_benzin["Week"].isin([29, 30, 31, 32]), "predict"] = new_benzin["8_month"]
new_benzin.loc[new_benzin["Week"].isin([33, 34, 35, 36]), "predict"] = new_benzin["9_month"]
new_benzin.loc[new_benzin["Week"].isin([37, 38, 39, 40]), "predict"] = new_benzin["10_month"]
new_benzin.loc[new_benzin["Week"].isin([41, 42, 43, 44]), "predict"] = new_benzin["11_month"]
new_benzin.loc[new_benzin["Week"].isin([45, 46, 47, 48]), "predict"] = new_benzin["12_month"]
new_benzin.loc[new_benzin["Week"].isin([49, 50, 51, 52]), "predict"] = new_benzin["13_month"]

last_benzin = new_benzin[["Customer", "predict"]]
last_benzin

##############
last_benzin
last_motorin

tablo = test[["index", "Customer", "Area"]]

motorin_tablo = tablo.loc[:54235, :]
benzin_tablo = tablo.loc[54236:,:]

mevcut_motorin = last_motorin[last_motorin["Customer"].isin(motorin_tablo["Customer"])]
mevcut_benzin = last_benzin[last_benzin["Customer"].isin(benzin_tablo["Customer"])]

###################
deneme = motorin_tablo.join(mevcut_motorin, lsuffix="left")
deneme.drop("Customer", axis=1, inplace=True)
deneme["Area"].value_counts()
deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "Akdeniz Bölgesi"), :] = deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "Akdeniz Bölgesi"), :].fillna(deneme[deneme["Area"] == "Akdeniz Bölgesi"]["predict"].mean())
deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "Marmara Bölgesi"), :] = deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "Marmara Bölgesi"), :].fillna(deneme[deneme["Area"] == "Marmara Bölgesi"]["predict"].mean())
deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "Ege Bölgesi"), :] = deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "Ege Bölgesi"), :].fillna(deneme[deneme["Area"] == "Ege Bölgesi"]["predict"].mean())
deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "İç Anadolu Bölgesi"), :] = deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "İç Anadolu Bölgesi"), :].fillna(deneme[deneme["Area"] == "İç Anadolu Bölgesi"]["predict"].mean())
deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "Karadeniz Bölgesi"), :] = deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "Karadeniz Bölgesi"), :].fillna(deneme[deneme["Area"] == "Karadeniz Bölgesi"]["predict"].mean())
deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "Güneydoğu Anadolu Bölgesi"), :] = deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "Güneydoğu Anadolu Bölgesi"), :].fillna(deneme[deneme["Area"] == "Güneydoğu Anadolu Bölgesi"]["predict"].mean())
deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "Doğu Anadolu Bölgesi"), :] = deneme.loc[(deneme["predict"].isnull()) & (deneme["Area"] == "Doğu Anadolu Bölgesi"), :].fillna(deneme[deneme["Area"] == "Doğu Anadolu Bölgesi"]["predict"].mean())


###############
deneme_benzin = benzin_tablo.join(mevcut_benzin, lsuffix="left")
deneme_benzin.drop("Customer", axis=1, inplace=True)
deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "Akdeniz Bölgesi"), :] = deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "Akdeniz Bölgesi"), :].fillna(deneme_benzin[deneme_benzin["Area"] == "Akdeniz Bölgesi"]["predict"].mean())
deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "Marmara Bölgesi"), :] = deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "Marmara Bölgesi"), :].fillna(deneme_benzin[deneme_benzin["Area"] == "Marmara Bölgesi"]["predict"].mean())
deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "Ege Bölgesi"), :] = deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "Ege Bölgesi"), :].fillna(deneme_benzin[deneme_benzin["Area"] == "Ege Bölgesi"]["predict"].mean())
deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "İç Anadolu Bölgesi"), :] = deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "İç Anadolu Bölgesi"), :].fillna(deneme_benzin[deneme_benzin["Area"] == "İç Anadolu Bölgesi"]["predict"].mean())
deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "Karadeniz Bölgesi"), :] = deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "Karadeniz Bölgesi"), :].fillna(deneme_benzin[deneme_benzin["Area"] == "Karadeniz Bölgesi"]["predict"].mean())
deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "Güneydoğu Anadolu Bölgesi"), :] = deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "Güneydoğu Anadolu Bölgesi"), :].fillna(deneme_benzin[deneme_benzin["Area"] == "Güneydoğu Anadolu Bölgesi"]["predict"].mean())
deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "Doğu Anadolu Bölgesi"), :] = deneme_benzin.loc[(deneme_benzin["predict"].isnull()) & (deneme_benzin["Area"] == "Doğu Anadolu Bölgesi"), :].fillna(deneme_benzin[deneme_benzin["Area"] == "Doğu Anadolu Bölgesi"]["predict"].mean())



deneme_benzin.isnull().sum()

submis2 = pd.concat([deneme, deneme_benzin])

x=submis2["predict"]

x.to_csv("submis2.csv")



deneme_benzin = benzin_tablo.join(mevcut_benzin,lsuffix="left")

deneme_benzin.drop("Customer", axis=1, inplace=True)

deneme_benzin.fillna(last_benzin["predict"].mean(), inplace=True)
deneme_benzin.drop("Customer", axis=1, inplace=True)
deneme_benzin



deneme.describe().T
deneme_benzin.describe().T

deneme
deneme_benzin