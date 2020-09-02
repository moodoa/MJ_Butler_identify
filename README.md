# MJ_Butler_identify
用 CNN 模型辨識 Michael Jordan 與 Jimmy Butler。
![alt text](https://a.espncdn.com/photo/2015/1112/nba_g_jordanbut_cr_1296x729.jpg)

## image_collector.py
* 以 `MichaelJordan`、`Jimmy Butler` 為參數蒐集 GOOGLE 上的圖片。
* 將蒐集回來的圖片做預處理，並儲存至 `training資料夾`、`test資料夾`。

## face_classifier.py
* 以 `training資料夾`、`test資料夾` 中的圖片為 training data 與 testing data 訓練模型。
* 回傳預測值，如圖。
![alt text](https://imgur.com/HJU05v9)

## Requirements
python 3

## Installation
`pip install -r requriements.txt`

