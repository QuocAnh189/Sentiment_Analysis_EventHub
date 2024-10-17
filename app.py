from flask import Flask, request, render_template
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def hello():
  return render_template("index.html")

@app.route('/add')
def add():
    a = int(request.args.get('a'))
    b = int(request.args.get('b'))
    return str(a + b)

if __name__== '__main__':
   app.run(port=8080, debug=True)
