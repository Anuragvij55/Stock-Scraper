import urllib.request, json , os, itertools
import pandas as pd
from multiprocessing.dummy import Pool
from datetime import datetime

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

def main():
	csv_path = os.getcwd()+os.sep+"historic_data"+os.sep+"csv"+os.sep
	if not os.path.isdir(csv_path):
		os.makedirs(csv_path)

	ticker_file_path ="Yahoo Ticker Symbols.xlsx"

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
	return

if __name__ == '__main__':
	main()