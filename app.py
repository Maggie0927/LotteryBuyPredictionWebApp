#!/usr/bin/env Python
# coding=utf-8

import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--port', type=int, default=8827)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--host', type=str, default='127.0.0.1')
parser.add_argument('--appname', type=str, default='')
parser.add_argument('--threaded', type=bool, default=True)
parser.add_argument('--access_logname', type=str, default='log/access.log')
args = parser.parse_args()

from flask import Flask, request
import json
import torch
import torch.nn.functional as F
from train import Net
from data_clean import input_to_label

app = Flask(__name__)
model = Net(input_size=18)
model.load_state_dict(torch.load('config/model.pkl'))
thres = 0.75

def error_messeage(error):
    ans = json.dumps({
        "code": 403,
        "message": f"{error} is invalid"
        })
    return ans

def ok_messeage(data):
    ans = json.dumps({
        "code": 200,
        "message": "ok",
        "data": data
    })
    return ans

@app.route('/sir/api/buy-predict', methods=['POST'])
def predict():
    input_feature = request.json
    beauty, emotion, expression, create_time = input_feature['beauty'], input_feature['emotion'], input_feature['expression'], input_feature['create_time']
    if not isinstance(beauty, int) or beauty > 100 or beauty < 0:
        return error_messeage('beauty')
    if not isinstance(emotion, str) or emotion == '':
        return error_messeage('emotion')
    if not isinstance(expression, str) or expression == '':
        return error_messeage('expression')
    if not str(create_time).isdigit() or len(str(create_time)) != 10:
        return error_messeage('create_time')

    features_list = [float(x) for x in input_to_label(input_feature)]
    features = torch.tensor([features_list], dtype=torch.float)
    outputs = model(features)
    res = F.softmax(outputs,dim=1).detach().cpu().numpy()[0]
    return ok_messeage({
        "nobuy_prob": float(res[0]),
        "haoyunshibei_prob": float(res[1]),
        "shuangseqiu_prob": float(res[2]),
        "fucai3d_prob": float(res[3]),
        "other_prob": float(res[4])
    })
    
if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=args.threaded)
    
