## Lottery Purchase Prediction Model

Objective: predict the lottery type that the user in the session will buy, using the discrete features from the user face image and user's historical purchase data.  

Goal: recommend lottery types to users and improve the order conversion rate, in order to increase sales revenue.

### Data Source

* The feature from the user face image in the Session (from Baidu Face Recognition API)： 
    
    Beauty, Expression, Emotion, Face ID (optional, only for old users)

* Other features from Session：
     
    Session Time (in 24 hours)

* Use the session face id to get user id, and retrieve the historical order data of this user： 
    
    City, Lottery Type, New/Old Users, Lottery Station Type (supermarket, restaurant), Total Purchase Days, Frequently Purchase Lottery Type
    
    If this is a new user and there is no user id, the feature from historical order data will be replaced by mean or mode.


### Data Cleaning and Selected Features

Transform the continuous variables to one-hot encoding variables, and check whether they are strongly correlated with the dependent variable. There are 18 features in total after variable selection:
    
  
|  Feature   | Source  |
|  ----  | ----  |
| Beauty  | session face |
| Laugh  | session face |
| Neutral Emotion  | session face |
| Positive Emotion  | session face |
| Session Time  | session |
| Yichang  | user attribute |
| Enshi  | user attribute |
| Wuhan  | user attribute |
| Ten Times Good Luck  | historical order |
| Qilecai  | historical order |
| Shilitaohua  | historical order |
| Other Lottery Type  | historical order |
| Total Purchase Days  | historical order |
| New User  | historical order |
| Clubhouse  | historical order |
| Chess Room  | historical order |
| Supermarket  | historical order |
| Restaurant  | historical order |

### Model Structure

Concatenate all the feaures, and input to a 3-layers MLP. Then perform a multiclass classification task and predict the lottery type the user will buy in the session (Two-color Ball, Ten Times Good Luck, Welfare Lottery 3D, Other Lottery Type)

### Prediction result using historical data

Accuracy metrics using the data from 07/2021:

| Type | Accuracy |
| ---- | ---- |
| Average Accuracy | 0.85 |
| No Buy | 0.833 |
| Ten Times Good Luck | 0.814 |
| Two-color Ball | 0.961 |
| Welfare Lottery 3D | 0.822 |
| Other Lottery Type | 0.908 |


### Model Call Method
```
python3 app.py \
    --port=8827 \
    --debug=False \
    --host='127.0.0.1' \
    --appname='buy_prediction' \
    --threaded=True
```
