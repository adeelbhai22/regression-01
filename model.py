import pandas as pd 
from sklearn.linear_model import LinearRegression
import pickle
df=pd.read_json('Data/House_Price.json')
X=df['Area(in sq. ft)'].values.reshape(-1,1)
y=df['Price(in Rs.)'].values.reshape(-1,1)
model =LinearRegression()
model.fit(X,y)
pickle.dump(model,open('model.pkl','wb'))