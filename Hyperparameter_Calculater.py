import math
from validate import validate
import matplotlib.pyplot as plt
from Train import *

def predict_target(test_X, model):
    node=model
    while node.left:
        if test_X[node.feature_index]<node.threshold :
            node=node.left
        else :
            node=node.right
    return node.predicted_class

            
def get_out_of_bag_error(models, train_XY, bootstrap_samples):
    oobe = 0
    num_of_elements_predicted = 0
    train_XY = train_XY.tolist()
    bootstrap_samples = bootstrap_samples.tolist()
    for train_elem in train_XY:
        num_bags_not_having_train_elem = 0
        misclassified_count = 0
        for bootstrap_sample in bootstrap_samples:
            if train_elem not in bootstrap_sample:
                index = list(bootstrap_samples).index(bootstrap_sample)
                model = models[index]
                x = train_elem[:-1]
                actual_y = train_elem[-1]
                predicted_y = predict_target(x,model)
                if(predicted_y != int(actual_y)):
                    misclassified_count += 1
                num_bags_not_having_train_elem += 1
        if(num_bags_not_having_train_elem > 0):
            oobe += (misclassified_count/float(num_bags_not_having_train_elem))
            num_of_elements_predicted += 1
            
    if(num_of_elements_predicted==0):
        return 0
    oobe /= float(num_of_elements_predicted)
    print(oobe)
    return oobe


def get_trained_models_using_bagging_of_hyper(bootstrap_samples,max_depth=60,min_size=1):
    models=list()
    for sample in bootstrap_samples:
        train_X=sample[:,:-1]
        train_Y=sample[:,-1]
        model=construct_tree(train_X,train_Y,max_depth,min_size,0)
        models.append(model)
    return models


def hypermaterKnowing_of_bootstrap_size(train_X, train_Y):
    #initaially take range from 5 to 50 at 5 step from that go to depth
    num_bootstrap_samples =[i for i in range(1,16)]
    train_XY=get_combined_train_XY(train_X, train_Y)
    list_of_oobr=[]
    list_of_num_bootstrape=[]
    for i in num_bootstrap_samples:
        random.seed(1)
        bootstrap_samples=get_bootstrap_samples(train_XY, i)
        models = get_trained_models_using_bagging_of_hyper(bootstrap_samples)
        oobr=get_out_of_bag_error(models, train_XY,bootstrap_samples)
        list_of_oobr.append(oobr)
        list_of_num_bootstrape.append(i)
    
    plt.plot(list_of_num_bootstrape,list_of_oobr)
    

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

    
def hypermaterKnowing_depth_of_tree(train_X,train_Y,validation_split_percent):
    length_training_Data_X=math.floor(((float(100-validation_split_percent))/100)*len(train_X))
    training_Data_X=train_X[0:length_training_Data_X]
    training_Data_Y=train_Y[0:length_training_Data_X]
    testing_Data_X = train_X[length_training_Data_X:]
    Actual_Data_Y = train_Y[length_training_Data_X:]
    max_depth=[i for i in range(1,len(train_X[0])+4)]
    min_size=[i for i in range(len(list(set(train_Y)))+2)]
    num_bootstrap_samples = 11
    train_XY=get_combined_train_XY(training_Data_X, training_Data_Y)
    for i in max_depth:
        for j in min_size:
            bootstrap_samples=get_bootstrap_samples(train_XY, num_bootstrap_samples)
            models = get_trained_models_using_bagging_of_hyper(bootstrap_samples,i,j)
            pred_Y = predict_target_values(testing_Data_X, models)
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(Actual_Data_Y, pred_Y)
            print(" this is accuracy for above max_depth---",i)
            print(" this is accuracy for above min_size---",j)
            print("Accuracy", accuracy)

    
    
    
X,Y=Import_data()
X=data_processing(X)
# Uncomment it for knowing better bootstrao size and depth,min_size of tree
#hypermaterKnowing_of_bootstrap_size(X,Y)
hypermaterKnowing_depth_of_tree(X,Y,30)