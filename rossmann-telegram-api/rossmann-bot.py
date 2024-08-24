import json
import requests
import pandas as pd
import os
from   flask             import Flask, request, Response

#constants
TOKEN = '6876015521:AAGQZyLyFQYh6s1tHMIZA6yAMA67OmBelL4'

# Info about the bot
# 'https://api.telKegram.org/bot6876015521:AAGQZyLyFQYh6s1tHMIZA6yAMA67OmBelL4/getMe'

# Get updates
# 'https://api.telegram.org/bot6876015521:AAGQZyLyFQYh6s1tHMIZA6yAMA67OmBelL4/getUpdates'

# Send message
# 'https://api.telegram.org/bot6876015521:AAGQZyLyFQYh6s1tHMIZA6yAMA67OmBelL4/sendMessage?chat_id=1049927447&text=Hi!'

# Webhook
# 'https://api.telegram.org/bot6876015521:AAGQZyLyFQYh6s1tHMIZA6yAMA67OmBelL4/setWebhook?url=https://da8c50fcd742f0.lhr.life'

def send_message( chat_id, text):

    url = 'https://api.telegram.org/bot{}/'.format(TOKEN)
    url= url + 'sendMessage?chat_id={}'.format(chat_id)

    r = requests.post(url, json={'text': text})
    print('Status Code {}'.format(r.status_code))

    return None


def load_dataset(store_id):

    # loading test dataset
    df11 = pd.read_csv('../data/test.csv')
    df_store_raw = pd.read_csv("../data/store.csv", low_memory=False)

    # merge test dataset + store
    df_test = pd.merge(df11, df_store_raw, how='left', on='Store')

    # choose store for prediction
    df_test = df_test[df_test['Store']== store_id]

    if not df_test.empty:

        # remove closed days
        df_test = df_test[(df_test['Open'].notnull()) & (df_test['Open'] !=0)]
        df_test = df_test.drop('Id', axis=1)

        # convert dataframe to json
        data = json.dumps(df_test.to_dict(orient='records'))

    else:
        data = 'error'

    return data

def predict(data):

    # API call
    # url = 'http://0.0.0.0:5000/rossmann/predict'
    url = 'https://ds-em-producao.onrender.com/rossmann/predict'
    header = {'Content-Type': 'application/json'}

    r = requests.post( url, data, headers=header)
    print('Status Code {}'.format(r.status_code))

    d1 = pd.DataFrame(r.json(), columns = r.json()[0].keys())

    return d1

# initialize API
app = Flask( __name__ )

def parse_message(message):

    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    store_id = store_id.replace('/', '')

    if store_id != 'start':

        try:
            store_id = int(store_id)
        except ValueError:
            store_id = 'error'

    return chat_id, store_id

@app.route( '/', methods=['GET', 'POST'] )
def index():
    if request.method == 'POST':
        message = request.get_json()
        chat_id, store_id = parse_message( message )
        
        if (store_id != 'error') & (store_id != 'start'):
            # loading data
            data = load_dataset(store_id)
            
            if data != 'error':

                # prediction
                d1 = predict(data)
                
                # calculation
                d2 = d1[['store', 'prediction']].groupby('store').sum().reset_index()

                # send message
                msg = 'Store Number {} will sell ${:,.2f} in the next 6 weeks'.format(
                        d2['store'].values[0],
                        d2['prediction'].values[0])
                send_message(chat_id, msg)

            else:
                send_message(chat_id, 'Store Not Found')

        elif store_id == 'start':
            send_message(chat_id, 'Hello, enter the number of the store you would like to forecast sales for the next 6 weeks.')
        else:
            send_message(chat_id, 'Store ID is Wrong')
    else:
        return '<h1> Rossmann Telegram BOT </h1>'
    return Response('Ok', status=200)

if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000) 
    app.run( host='0.0.0.0', port = port )

