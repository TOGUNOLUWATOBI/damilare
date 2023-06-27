from flask import Flask, request, jsonify
from predictor import predict_symptom_knn, predict_symptom_svm, predict_neural_network
app = Flask(__name__)
from flask_cors import CORS, cross_origin



@app.route('/symptom_predictor', methods=['GET','POST'])
@cross_origin()
def symptom_predictor():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'No data provided'})
    # Process the array as needed
    symptom_array = [data['arr']]
    ANN_pred = predict_neural_network(symptom_array)
    SVM_pred = predict_symptom_svm(symptom_array)
    KNN_pred = predict_symptom_knn(symptom_array)
    return jsonify({'ANN_pred': ANN_pred[0],'SVM_pred':SVM_pred[0],"KNN_pred":KNN_pred[0]})


if __name__ == 'main':
    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    app.run()