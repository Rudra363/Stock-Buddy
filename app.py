from flask import Flask, request, jsonify, render_template
import io
import sys
from main import main

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('StockBuddyGUI.html')

def run_main_and_capture_output(ticker):
    captured_output = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        main(ticker)
    except Exception as e:
        print(f"Error during prediction: {e}")
    finally:
        sys.stdout = original_stdout  
    return captured_output.getvalue()



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker', '').strip()
    if not ticker:
        return jsonify({"result": "Invalid ticker provided."}), 400

    output = run_main_and_capture_output(ticker)
    return jsonify({"result": output})


if __name__ == '__main__':
    app.run(debug=True)
