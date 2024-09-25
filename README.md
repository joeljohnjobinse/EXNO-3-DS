## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/fadad811-94d4-4545-ab80-f21b02f5e7b7)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/809ffbf8-3ff1-4aec-911f-94f34d36a2b5)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/54a34281-a0e8-47dd-9655-7a70735ff77d)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/2cc0515d-82a6-4ecc-80e3-9d7e7d61b4ee)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![image](https://github.com/user-attachments/assets/c0519e0c-e96d-47fa-ac90-3bfe9b6f44d1)

```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/e933a5c4-2a13-4584-8ef3-885569d72697)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/5de13ed8-465b-4600-87c8-fc514aa5443b)

```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/dc82fe47-6a14-4fa8-81a1-0d323201ebeb)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/073af0f7-c94e-4af5-bfba-e54a7e9e677f)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/2e0e2100-693e-4fa9-a1b8-1fe8018118c6)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/8119e3c2-256a-4087-bf0b-2c5379855dd0)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/6a4fb3f6-bbee-459a-b1b4-41be88d722de)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/994176b4-17b1-41b1-9cf4-0d51fb5cfb25)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/bfa03fcf-37cb-4ed4-9180-f4ea07ba7886)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/c2bccbd8-d289-4661-8254-7ada08bf8dc8)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/479c1260-747d-41dc-8735-cbe47d57eb60)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/51d4d8ff-cfd0-469f-ab54-714d77651413)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/6d70b042-60e0-4fae-8b6b-5773eb22aa72)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
![image](https://github.com/user-attachments/assets/f23d3d00-a23d-409c-9e57-7935ca8b0354)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/d4a824f7-8bc3-4121-a659-f50872251322)

```
df["Highly Negative Skew_yeo-johnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/3ead7dd1-930a-4adc-b2ae-f79efd137754)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])

df
```
![image](https://github.com/user-attachments/assets/6bab3e60-904c-4ad3-a641-866f19158e5c)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/07bee11b-9c7b-49c9-8a45-e247ef135e10)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/ce9b5d61-0925-4783-b1ca-60247df4f990)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/672eb546-6ad8-4658-be39-22a356de8f69)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/7b0b8603-502d-4a99-aa02-d0933e35b059)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/444b094a-7b84-42a9-86b2-c374d63676bf)


# RESULT:
Thus, performing Feature Encoding and Transformation process for the given data set is completed.

       
