import requests
import ssl
import time
import hashlib
import json as js
import os
from datetime import datetime
from urllib import parse
import xml.etree.ElementTree as ET

def dict2list(dic:dict):
    ''' 将字典转化为列表 '''
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst

def curlmd5(src):
    m = hashlib.md5()
    m.update(src.encode('UTF-8'))
    return m.hexdigest()


def get_data(url = 'http://api.cngrid.org/v2/show/cngrid/realtimeInfo'):
    ssl._create_default_https_context = ssl._create_unverified_context
    values = {'username': 'xulubuaa', 'password': 'xulu416','appid':'opus','remember':'true'}
    s = requests.Session()
    r = s.post("https://api.cngrid.org/v2/users/login",values,verify=False)
    root = ET.fromstring(r.text)
    logdic = {}
    for child in root:
        logdic[child.tag] = child.text

    #签名算法的计算
    params = {}
    millis = int(round(time.time() * 1000))
    params['timestamp'] = str(millis)
    md5dic_sort = sorted(dict2list(params), key=lambda x:x[0], reverse=False)
    md_str = ""
    for i in md5dic_sort:
        md_str= md_str+i[0]+'='+i[1]
    md_str+=logdic['md5secret']
    HTTP_METHOD = "GET"
    md_str = HTTP_METHOD+url+md_str
    md5 = curlmd5(md_str)
    params['md5sum'] = str(md5)
    params_data = parse.urlencode(params).encode('utf-8')  # 提交类型不能为str，需要为byte类型
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json','User-Agent':"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"}

    r = s.get(url, params=params_data,headers=headers,verify=False)
    return r.json()
