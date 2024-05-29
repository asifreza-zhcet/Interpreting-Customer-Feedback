from flask import Flask, render_template, request
from main import predict_senti
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    if request.method == 'POST':
        data = request.form['inputext']
        prediction = predict_senti(data)[0]
        print(prediction)
        return render_template('predict.html', item=prediction)
    return render_template('base.html')

if __name__ == '__main__':
    app.run(debug=True)

