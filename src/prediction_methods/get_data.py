import websocket, json
import csv

cc = 'btcusdt'
interval = '3m'
socket = f'wss://stream.binance.com:9443/ws/{cc}@kline_{interval}'

ws = websocket.WebSocket()
list_data = []

def get_data(ws, list_data):
    timer = 0
    ws.connect(socket)

    while(timer < 5):
        data = ws.recv()
        data = data.split('k', 1)
        data = data[1]
        data = '{' + data[43:len(data) - 2] + '}'
        data = json.loads(data)
        list_data.append([data["h"], data["l"], data["c"]])
        timer += 1

    if (timer == 5):
        ws.close()

get_data(ws, list_data)

def write_data(list_data):
    csvFile = open('./data/processed_3minutes.csv', 'w', newline='', encoding='UTF8')
    candleStickWriter = csv.writer(csvFile,delimiter=',')
    candleStickWriter.writerow(['High','Low','Close'])

    for i in range(len(list_data)):
        candleStickWriter.writerow(list_data[i])

    csvFile.close()

write_data(list_data)


