import pandas as pd
from flask import Flask, jsonify, request
import pickle
import cv2
from keras import models
import numpy as np

# def process(str):
#     str = str.replace('[', '')
#     str = str.replace(']', '')
#     str = str.split(', ')
#     lst = list(map(int, str))
#     #print(len(lst))
#     arr = np.array(lst)
#     arr = arr.reshape(200, 200, 3)
#     #print(type(arr[0,0][0]), type(img1[0,0][0]))
#     #print((img1 == arr).all())
#     arr = arr.astype('uint8')
#     image = cv2.resize(arr, (64, 64))
#     image = np.array(image)
#     image = image.astype('float32')/255.0
#     image = image.reshape(-1, 64, 64, 3)
#     return image

# load model
model = pickle.load(open('model.pkl','rb'))
my_model = models.load_model('ASL1.h5')

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)
    
#     image = process(data)
#     my_model.predict(image)
#     x = np.argmax(my_model.predict(image), axis=1)
#     # convert data into dataframe
#     data.update((x, [y]) for x, y in data.items())
#     data_df = pd.DataFrame.from_dict(data)

#     # predictions
#     result = model.predict(data_df)

#     # send back to browser
#     output = {'results': int(result[0])}

#     # return data
#     return jsonify(results=output)

    return jsonify(data)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

