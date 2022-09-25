# Пример запуска: run test_forecast.py "DS_train(2020-06--2022-06-01).csv" "learn" "Y_train.xlsx"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import argparse

# импорт моделей
import xgboost as xg

# импорт метрики
from sklearn.metrics import mean_absolute_error

def run_neural_network(path: str, is_forecast: str, target_path: str):
    # загружаем данные, устанавливаем месячный период
    ds=pd.read_csv(path, sep='\t', index_col='DateObserve')
    ds.index = pd.to_datetime(ds.index).to_period('M')

    # Убираем из данных пропуски, дубликаты, заполняем до однородных временных меток
    ds.drop_duplicates(subset=['WebPriceId','CurrentPrice'],keep= False, inplace=True)
    ds = ds[['WebPriceId','CurrentPrice']]
    ds = ds.dropna()
    ds = ds.reset_index()
    ds = ds.drop_duplicates(subset=['DateObserve','WebPriceId'], keep='last')
    ds=ds.set_index(['DateObserve','WebPriceId']).unstack(fill_value=0).asfreq('M', fill_value=0).stack().sort_index(level=1).reset_index()
    ds.CurrentPrice = ds.CurrentPrice.replace(to_replace=0, method ='ffill')
    ds = ds[(ds != 0).all(1)]

    #Расчет множителя ИПЦ
    ds['CPI_multi'] = ds['CurrentPrice'].iloc[-1] / ds['CurrentPrice']
    df = ds.groupby('DateObserve')['CPI_multi'].mean()

    df = df.to_frame(name='cpi_multi')

    if (is_forecast =="learn"):
      # загружаем таргет 

      data = pd.read_excel(target_path, index_col='ИПЦ, мом')
      data = data.T
      data = data.set_index('Период')
      data = data[['Целевой показатель']]
      data.index = pd.to_datetime(data.index).to_period('M')
      # соединяем таблицы
      ds = df.join(data, how='outer')

      """ разделяем на тестовую и тренировочную выборки"""

      train, test = ds[0:(len(data)-1)], ds[(len(data)-1):]

      # Делим выборки на таргет и фичи
      y_train= train['Целевой показатель']
      x_train = train.drop(labels = ['Целевой показатель'], axis=1)

      y_test = test['Целевой показатель']
      x_test = test.drop(labels = ['Целевой показатель'], axis=1)


      model = xg.XGBRegressor(objective ='reg:squarederror',
                      n_estimators = 10, seed = 123)

      # Фит
      model.fit(x_train, y_train)
      
      # Предсказание
      pred = model.predict(x_test)
      #mean_absolute_error(test['Целевой показатель'], pred)
      model.save_model("model.json")
 
    # сохраняем и возвращаем результат в виде таблицы

      ds = ds.reset_index()
      y = y_train.reset_index()
      y = np.array(y[['Целевой показатель']])
      y = np.append(y,pred[0])

      tdf = pd.DataFrame({'период':np.array(ds.DateObserve), 'целевая функция':y})

      tdf.to_json(f'result.json',orient = 'split',index = False)
      #return tdf.to_json(f'result.json',orient = 'split',index = False)


if __name__ == "__main__":
    """получение пути к файлу и имени датасета"""
    parser = argparse.ArgumentParser(description='Передайте путь к датасету , слово learn и путь к целевым функциям')
    parser.add_argument('path', type=str)
    parser.add_argument('forecast', type=str)
    parser.add_argument('target_path', type=str)
    args = parser.parse_args()

    path = args.path
    is_forecast = args.forecast # = "learn"
    target_path = args.target_path

    run_neural_network(path, is_forecast, target_path)
