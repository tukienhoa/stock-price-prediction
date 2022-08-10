import csv
import config.config as cfg
from binance.client import Client
from datetime import timedelta, datetime


client = Client(cfg.API_KEY, cfg.API_SECRET, tld='us')

def getData():
    processed_data = []
    received_data = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, limit = 1000)
    for data in received_data:
        dateTime = datetime.fromtimestamp(data[0] / 1000 + timedelta(hours = 7).total_seconds())
        candlestick = {
            "date": dateTime.strftime("%d-%m-%Y %H:%M:%S"),
            "high": data[2],
            "low": data[3],
            "close": data[4]
        }
        processed_data.append(candlestick)

    return processed_data

def writeData(list_data):
    csvFile = open('./data/processed_1minute.csv', 'w', newline='', encoding='UTF8')
    candleStickWriter = csv.writer(csvFile,delimiter=',')
    candleStickWriter.writerow(['Date','High','Low','Close'])

    for i in range(len(list_data)):
        candleStickWriter.writerow([list_data[i]["date"], list_data[i]["high"], list_data[i]["low"], list_data[i]["close"]])

    csvFile.close()

def addData(data):
    csvFile = open('./data/processed_1minute.csv', 'a', newline='', encoding='UTF8')
    candleStickWriter = csv.writer(csvFile)
    candleStickWriter.writerow([data["date"], data["high"], data["low"], data["close"]])

    csvFile.close()

# writeData(processed_data)

