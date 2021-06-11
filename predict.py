import numpy as np
import csv
import sys
import pickle
from Train import Node
from validate import validate

mean_null_list=[[0.32899615, 0.03675128, 0.30589487, 0.27599423, 0.16040833,0.00630769, 0.37946731, 0.64194423, 0.20980385, 0.71879359, 0.35354744, 0.01347372, 0.71182949, 0.01077692, 0.04199936,0.00789103, 0.39709487, 0.50816218, 0.01649551, 0.00777949,0.01025 , 0.0276359 , 0.42374679, 0.68095449, 0.67020256,0.55927308, 0.19912564, 0.31843782, 0.12306282, 0.28124359,0.21726923, 0.07651667, 0.0202109 , 0.05283013, 0.13550513,0.61906795, 0.25237564, 0.28759679, 0.12346731, 0.37805192,0.00816859, 0.09395577, 0.71476154, 0.29619103, 0.60308269,0.42222949, 0.25283654, 0.10645513, 0.5152641 , 0.23770385,0.44666987, 0.05339231, 0.37261538, 0.66015577, 0.31837308,0.59345449, 0.00741282, 0.17668974, 0.46012628, 0.00923462]]

All_mini_maxi_mean_normalize_value=[[0.0383, 1.0, 0.3289961538461538], [0.0006, 0.1632, 0.03675128205128205], [0.0117, 0.9297, 0.3058948717948718], [0.0493, 0.7022, 0.27599423076923074], [0.0025, 0.7292, 0.16040833333333332], [0.0006, 0.0439, 0.006307692307692308], [0.0223, 1.0, 0.3794673076923077], [0.0563, 1.0, 0.6419442307692308], [0.0193, 0.7106, 0.20980384615384615], [0.0481, 1.0, 0.7187935897435898], [0.0351, 0.9497, 0.3535474358974359], [0.0008, 0.0709, 0.013473717948717946], [0.0921, 1.0, 0.7118294871794871], [0.0005, 0.039, 0.010776923076923076], [0.0015, 0.1997, 0.041999358974358976], [0.0003, 0.044, 0.007891025641025642], [0.0401, 0.9647, 0.3970948717948718], [0.0746, 1.0, 0.5081621794871796], [0.0015, 0.1004, 0.016495512820512818], [0.0007, 0.0355, 0.007779487179487179], [0.001, 0.0352, 0.01025], [0.0025, 0.1083, 0.027635897435897432], [0.0349, 1.0, 0.4237467948717949], [0.0747, 1.0, 0.6809544871794873], [0.0239, 1.0, 0.6702025641025642], [0.0656, 1.0, 0.5592730769230769], [0.0095, 0.7034, 0.19912564102564104], [0.0031, 1.0, 0.3184378205128205], [0.0073, 0.5522, 0.12306282051282051], [0.0056, 0.8246, 0.2812435897435897], [0.0255, 0.7762, 0.21726923076923077], [0.0067, 0.3225, 0.07651666666666666], [0.0006, 0.0825, 0.020210897435897435], [0.0058, 0.2604, 0.05283012820512822], [0.0057, 0.459, 0.1355051282051282], [0.0689, 1.0, 0.6190679487179488], [0.0308, 0.7733, 0.25237564102564103], [0.0386, 0.8995, 0.28759679487179485], [0.0033, 0.3729, 0.12346730769230771], [0.0162, 0.9988, 0.37805192307692304], [0.0006, 0.0394, 0.008168589743589744], [0.0041, 0.3339, 0.09395576923076925], [0.0598, 1.0, 0.7147615384615386], [0.0336, 0.997, 0.2961910256410257], [0.0512, 1.0, 0.6030826923076923], [0.0477, 0.9708, 0.4222294871794872], [0.0236, 0.706, 0.2528365384615385], [0.0116, 0.3823, 0.10645512820512817], [0.0482, 0.9657, 0.5152641025641026], [0.0327, 0.7342, 0.23770384615384615], [0.0404, 0.9306, 0.4466698717948718], [0.0073, 0.1981, 0.05339230769230769], [0.008, 1.0, 0.3726153846153846], [0.0408, 1.0, 0.6601557692307692], [0.0371, 0.9709, 0.3183730769230769], [0.0823, 1.0, 0.5934544871794872], [0.0001, 0.0332, 0.007412820512820513], [0.0075, 0.6587, 0.1766897435897436], [0.0837, 1.0, 0.4601262820512822], [0.0006, 0.0447, 0.009234615384615385]]


def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    model = pickle.load(open(model_file_path, 'rb'))
    return test_X, model

def replace_null_values_with_mean(X):
    mean_of_nan=mean_null_list[0]
    index=np.where(np.isnan(X))
    X[index]=np.take(mean_of_nan,index[1])
    return X

def mean_normalize(X, column_indices):
    mini,maxi,mean=All_mini_maxi_mean_normalize_value[column_indices]
    X[:,column_indices]=(X[:,column_indices]-mean)/(maxi-mini)
    return X

def data_processing(class_X) :
    X=replace_null_values_with_mean(class_X)
    for i in range(class_X.shape[1]):
        X=mean_normalize(X,i)
    
    return X

def predict_target(test_X, model):
    node=model
    while node.left:
        if test_X[node.feature_index]<node.threshold :
            node=node.left
        else :
            node=node.right
    return node.predicted_class


def predict_target_values(test_X, models):
    predicted_value=list()
    for data in test_X:
        predictions = [predict_target(data,model) for model in models]
        classes = sorted(list(set(predictions)))
        max_voted_class = -1
        max_votings = -1
        for c in classes:
            if(predictions.count(c) > max_votings):
                max_voted_class = c
                max_votings = predictions.count(c)

        predicted_value.append(max_voted_class)
        
    return np.array(predicted_value)


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    test_X=data_processing(test_X)
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_rf.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_rf.csv") 