import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('marketing_campaign_dataset.csv')
df['Acquisition_Cost'] = df['Acquisition_Cost'].str.strip('$').str.replace(',', '').astype(float)
df['Duration'] = df['Duration'].str.extract('(\d+)').astype(float)
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df.drop(columns=['Date'], inplace=True)
categorical_columns = ['Company', 'Campaign_Type', 'Target_Audience', 'Channel_Used',
                       'Location', 'Language', 'Customer_Segment']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
X = df.drop(columns=['ROI'])
y = df['ROI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=7)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mse, r2)

# model = LinearRegression()
# model.fit(X=df[['Acquisition_Cost']], y=df['ROI'])
# model.coef_
# df = df[0:200]
# fig = px.scatter(df, x='Acquisition_Cost', y='ROI')
# fig.add_trace(go.Scatter(
#     x=df['Acquisition_Cost'],
#     y=model.predict(df[['Acquisition_Cost']]),
#     mode='lines',
#     name='Linear Model'
# ))
# fig.update_layout(title='Test',
#                   xaxis_title='test', 
#                   yaxis_title='test')
# fig.show()
# print(model.coef_)