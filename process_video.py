from ultralytics import YOLO
import pandas as pd
import numpy as np
import json


def process_video(video_path):
    from ultralytics import YOLO
    model_path = '/home/jupyter/datasphere/project/yolov8l.engine'
    tracker_path = '/home/jupyter/datasphere/project/bytetrack.yaml'

    model = YOLO(model_path)
    results = model.track(source=video_path,
                          tracker=tracker_path,
                          imgsz=544,
                          stream=True,
                          verbose=True,
                          persist=True)

    r = {
        'id': [],
        'cls': [],
        'xywh': [],
        'xywhn': [],
        'xyxy': [],
        'xyxyn': [],
        'orig_shape': [],
    }

    ln = 0
    for x in results:
        r['id'].append(x.boxes.id.tolist() if not isinstance(x.boxes.id, type(None)) else [-1]*len(x.boxes.xywh))
        r['cls'].append(x.boxes.cls.tolist() if not isinstance(x.boxes.cls, type(None)) else [-1]*len(x.boxes.xywh))
        r['xywh'].append(x.boxes.xywh.tolist() if not isinstance(x.boxes.xywh, type(None)) else [-1]*len(x.boxes.xywh))
        r['xywhn'].append(x.boxes.xywhn.tolist() if not isinstance(x.boxes.xywhn, type(None)) else [-1]*len(x.boxes.xywh))
        r['xyxy'].append(x.boxes.xyxy.tolist() if not isinstance(x.boxes.xyxy, type(None)) else [-1]*len(x.boxes.xywh))
        r['xyxyn'].append(x.boxes.xyxyn.tolist() if not isinstance(x.boxes.xyxyn, type(None)) else [-1]*len(x.boxes.xywh))
        r['orig_shape'].append(x.orig_shape)
        ln += 1

    x_mnozh = 1
    s = 60 * 20 + 40
    ss = ln / s / x_mnozh

    df = pd.DataFrame(r)
    df = df.explode(['id', 'cls', 'xywh', 'xywhn', 'xyxy', 'xyxyn'])

    df = df.dropna()
    df = df[df['cls'] != 0]
    df = df[df['id'] != -1]
    file_name = video_path.split('/')[-1].split('.')[0]
    p = f'/home/jupyter/datasphere/project/TEST/markup/{file_name}.json'
    with open(p, 'r') as f:
        js = json.load(f)
    area = js['areas']

    df['x'] = df['xywhn'].apply(lambda x: x[0])
    df['y'] = df['xywhn'].apply(lambda x: x[1])
    df['y_or'] = df['orig_shape'].apply(lambda x: x[0])
    df['x_or'] = df['orig_shape'].apply(lambda x: x[1])
    df['xlu'] =  min([x[0][0] for x in area ])
    df['ylu'] =  min([x[0][1]  for x in area ])
    df['xru'] =  max([x[1][0]  for x in area ])
    df['yru'] =  min([x[1][1]  for x in area ])
    df['xrd'] =  max([x[2][0]  for x in area ])
    df['yrd'] =  max([x[2][1]  for x in area ])
    df['xld'] =  min([x[3][0]  for x in area ])
    df['yld'] =  max([x[3][1] for x in area ])

    _x1lu = min([x[0][0] for x in js['zones'][::2]])
    _y1lu = min([x[0][1] for x in js['zones'][::2]])
    _x1ld = min([x[1][0] for x in js['zones'][::2]])
    _y1ld = max([x[1][1] for x in js['zones'][::2]])
    _x1rd = max([x[2][0] for x in js['zones'][::2]])
    _y1rd = max([x[2][1] for x in js['zones'][::2]])
    _x1ru = max([x[3][0] for x in js['zones'][::2]])
    _y1ru = min([x[3][1] for x in js['zones'][::2]])

    _x2lu = min([x[0][0] for x in js['zones'][1::2]])
    _y2lu = min([x[0][1] for x in js['zones'][1::2]])
    _x2ld = min([x[1][0] for x in js['zones'][1::2]])
    _y2ld = max([x[1][1] for x in js['zones'][1::2]])
    _x2rd = max([x[2][0] for x in js['zones'][1::2]])
    _y2rd = max([x[2][1] for x in js['zones'][1::2]])
    _x2ru = max([x[3][0] for x in js['zones'][1::2]])
    _y2ru = min([x[3][1] for x in js['zones'][1::2]])


    df_cls_fact = pd.DataFrame(df.groupby(['id'])['cls'].agg(pd.Series.mode)).reset_index()
    df_cls_fact.columns=['id', 'cls_fact']
    df_cls_fact['cls_fact'] = df_cls_fact['cls_fact'].apply(lambda x: x[0] if isinstance(x, type(np.array([1]))) else x)
    df = df.merge(df_cls_fact, how='inner', on='id')

    df_res = df[
        (((df['x'] >= df['xlu'])&(df['x'] <= df['xru']) | (df['x'] >= df['xld'])&(df['x'] <= df['xrd']))\
        & ((df['y'] >= df['ylu'])&(df['y'] <= df['yld']) | (df['y'] >= df['yru'])&(df['y'] <= df['yrd'])))
    ]

    c = df_res.groupby('cls_fact')['id'].nunique()
    c = c.to_dict()

    df_res_speed_1 = df[(((df['x'] >= _x1lu) & (df['x'] <= _x1ru)) | ((df['x'] >= _x1ld) & (df['x'] <= _x1rd))\
        & (((df['y'] >= _y1lu) & (df['y'] <= _y1ld)) | ((df['y'] >= _y1ru) & (df['y'] <= _y1rd))))]
    df_res_speed_2 = df[(((df['x'] >= _x2lu) & (df['x'] <= _x2ru)) | ((df['x'] >= _x2ld) & (df['x'] <= _x2rd))\
        & (((df['y'] >= _y2lu) & (df['y'] <= _y2ld)) | ((df['y'] >= _y2ru) & (df['y'] <= _y2rd))))]
    valid_id = list(set(df_res_speed_1['id'].unique()) & set(df_res_speed_2['id'].unique()))

    s = ((20 / 1000) / (df_res[df_res['id'].isin(valid_id)].groupby(['id', 'cls_fact'])['cls'].count() / ss / 3600)).reset_index().groupby(['cls_fact'])['cls'].mean()
    s = s.to_dict()

    file_name = video_path.split('/')[-1]
    result = [file_name, c.get(2, 0), s.get(2, 0), c.get(7, 0), s.get(7, 0), c.get(5, 0), s.get(5, 0)]
    
    return result