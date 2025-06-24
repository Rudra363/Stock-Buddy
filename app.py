from flask import Flask, request, jsonify
from main import main  # Replace with your main module's name

import io
import sys

app = Flask(__name__)

def run_main_and_capture_output(ticker):
    # Redirect stdout to capture print output from main()
    captured_output = io.StringIO()
    sys.stdout = captured_output
    try:
        main(ticker)  # Call your main() with the provided ticker
    except Exception as e:
        print(f"Error during prediction: {e}")
    sys.stdout = sys.stdout  # Reset stdout
    return captured_output.getvalue()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker', '').strip()
    if not ticker:
        return jsonify({"result": "Invalid ticker provided."}), 400

    output = run_main_and_capture_output(ticker)
    return jsonify({"result": output})
#
if __name__ == '__main__':
    app.run(debug=True)