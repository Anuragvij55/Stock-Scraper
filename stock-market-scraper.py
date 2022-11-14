import urllib.request, json , os, itertools
import pandas as pd
from multiprocessing.dummy import Pool
from datetime import datetime
from os import listdir
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
import datetime as dt
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

def get_historic_price(query_url,csv_path):
	
	stock_id=query_url.split("&period")[0].split("symbol=")[1]

	if os.path.exists(csv_path+stock_id+'.csv') and os.stat(csv_path+stock_id+'.csv').st_size != 0:
		print("Historical data of "+stock_id+" already exists")
		return

	try:
		with urllib.request.urlopen(query_url) as url:
			parsed = json.loads(url.read().decode())
	
	except:
		print("Historical data of "+stock_id+" doesn't exist")
		return
	
	else:	
		try:
			Date=[]
			for i in parsed['chart']['result'][0]['timestamp']:
				Date.append(datetime.utcfromtimestamp(int(i)).strftime('%d-%m-%Y'))

			Low=parsed['chart']['result'][0]['indicators']['quote'][0]['low']
			Open=parsed['chart']['result'][0]['indicators']['quote'][0]['open']
			Volume=parsed['chart']['result'][0]['indicators']['quote'][0]['volume']
			High=parsed['chart']['result'][0]['indicators']['quote'][0]['high']
			Close=parsed['chart']['result'][0]['indicators']['quote'][0]['close']
			Adjusted_Close=parsed['chart']['result'][0]['indicators']['adjclose'][0]['adjclose']

			df=pd.DataFrame(list(zip(Date,Low,Open,Volume,High,Close,Adjusted_Close)),columns =['Date','Low','Open','Volume','High','Close','Adjusted Close'])

			if os.path.exists(csv_path+stock_id+'.csv'):
				os.remove(csv_path+stock_id+'.csv')
			df.to_csv(csv_path+stock_id+'.csv', sep=',', index=None)
			print("Historical data of "+stock_id+" saved")
		
		except:
			print("Historical data of "+stock_id+" could not be saved")
		return
def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def main():
	csv_path = os.getcwd()+os.sep+".."+os.sep+"historic_data"+os.sep+"csv"+os.sep
	if not os.path.isdir(csv_path):
		os.makedirs(csv_path)

	ticker_file_path = "Assets"+os.sep+"Yahoo Ticker Symbols small.xlsx"

	temp_df = pd.read_excel(ticker_file_path)

	headers = temp_df.iloc[2]
	df  = pd.DataFrame(temp_df.values[3:], columns=headers)
	print("Total stocks:",len(df))

	query_urls=[]

	for ticker in df['Ticker']:
		query_urls.append("https://query1.finance.yahoo.com/v8/finance/chart/"+ticker+"?symbol="+ticker+"&period1=0&period2=9999999999&interval=1d&includePrePost=true&events=div%2Csplit")

	with Pool(processes=10) as pool:
		pool.starmap(get_historic_price, zip(query_urls, itertools.repeat(csv_path)))

	print("Historical data of all stocks saved")
	csv_files=find_csv_filenames(csv_path)
	for file in csv_files:
		read_df = pd.read_csv(csv_path+os.sep+file)
		read_df.set_index("Date", inplace=True)
		read_df['Adjusted Close'].plot()
		plt.ylabel("Adjusted Close Prices")
		plt.show()
		df = pd.read_csv(csv_path+os.sep+file)
		df.set_index("Date", inplace=True)
		df.dropna(inplace=True)
		x = df.iloc[:, 0:5].values
		y = df.iloc[:, 4].values
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.26,  random_state=0)
		scale = StandardScaler()
		best_param=[]
		mean_absolute=[]
		mean_square=[]
		root_mean_square=[]
		R2_score=[]
		train_score=[]
		test_score=[]
		accur=[]
		x_train = scale.fit_transform(x_train)
		x_test = scale.transform(x_test)
		model = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_split=2, min_samples_leaf=1, max_depth=10, bootstrap=True)
		model.fit(x_train, y_train)
		predict = model.predict(x_test)
		grid_rf = {
		'n_estimators': [20, 50, 100, 500, 1000],  
		'max_depth': np.arange(1, 15, 1),  
		'min_samples_split': [2, 10, 9], 
		'min_samples_leaf': np.arange(1, 15, 2, dtype=int),  
		'bootstrap': [True, False], 
		'random_state': [1, 2, 30, 42]
		}
		rscv = RandomizedSearchCV(estimator=model, param_distributions=grid_rf, cv=3, n_jobs=-1, verbose=2, n_iter=200)
		rscv_fit = rscv.fit(x_train, y_train)
		# best_parameters = rscv_fit.best_params_
		best_param.append(rscv_fit.best_params_)
		mean_absolute.append(round(metrics.mean_absolute_error(y_test, predict), 4))
		mean_square.append(round(metrics.mean_squared_error(y_test, predict), 4))
		root_mean_square.append(round(np.sqrt(metrics.mean_squared_error(y_test, predict)), 4))
		R2_score.append(round(metrics.r2_score(y_test, predict), 4))
		train_score.append(model.score(x_train, y_train) * 100)
		test_score.append(model.score(x_test, y_test) * 100)
		# print(best_param)
		# print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, predict), 4))
		# print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, predict), 4))
		# print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, predict)), 4))
		# print("(R^2) Score:", round(metrics.r2_score(y_test, predict), 4))
		# print(f'Train Score : {model.score(x_train, y_train) * 100:.2f}% and Test Score : {model.score(x_test, y_test) * 100:.2f}% using Random Tree Regressor.')
		errors = abs(predict - y_test)
		mape = 100 * (errors / y_test)
		accuracy = 100 - np.mean(mape)
		accur.append(round(accuracy, 2))
		# print('Accuracy:', round(accuracy, 2), '%.') 
		startDate = dt.datetime.strptime(df.index[-1] ,'%d-%m-%Y')
		ind=pd.date_range(start=startDate, periods=len(predict), freq="D")
		predictions = pd.DataFrame({"Date":ind,"Predictions": predict} )
		predictions.to_csv("Predicted-price-data.csv")
		#colllects future days from predicted values
		df = pd.DataFrame(predictions[:21])
		df.to_csv("one-month-predictions.csv")
		print(df.head())
		onemonth_df_pred = pd.read_csv("one-month-predictions.csv")
		onemonth_df_pred.set_index("Date", inplace=True)
		buy_price = min(onemonth_df_pred["Predictions"])
		sell_price = max(onemonth_df_pred["Predictions"])
		onemonth_buy = onemonth_df_pred.loc[onemonth_df_pred["Predictions"] == buy_price]
		onemonth_sell = onemonth_df_pred.loc[onemonth_df_pred["Predictions"] == sell_price]
		print("Buy price and date")
		print(onemonth_buy)
		print("Sell price and date")
		print(onemonth_sell)
		onemonth_df_pred["Predictions"].plot(figsize=(10, 5), title="Forecast for the next 1 month", color="blue")
		plt.xlabel("Date")
		plt.ylabel("Price")
		plt.legend()
		plt.show()
	return





if __name__ == '__main__':
	main()