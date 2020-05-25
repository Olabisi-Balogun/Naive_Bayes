def get_data():
    file = open("animal2.txt","r")

    raw_data = []

    for line in file:
        raw_data.append(line.strip("\n"))
    attributes = raw_data[0]

    
    data = []
    test_data = []
    for i in raw_data:
        data.append(list(i.split(",")))
        
    animal_names = []
    for k in data:
        name = k.pop(0)
        animal_names.append(name)
    
    data = data[1:]
    
        
    for j in range(len(data)):
        if data[j][-1]== '-1':
            test_data.append(data[j])
    
    train_data = [train for train in data if train[-1]!= '-1']
    
    
    return train_data,test_data

def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

def get_class_probabilities(separated, train):
    class_probabilities = dict()
    for class_label in separated:
        class_probabilities[class_label] = (len(separated[class_label])+0.1)/float((len(train)+0.1)+(0.1*len(separated)))
        
    return class_probabilities       

def get_class_cond_probabilities(test,rows):
    unique_f = [2,2,2,2,2,2,2,2,2,2,2,2,6,2,2,2]
    attr_prob =1.0
    all_attr_prob = 1
    for i in range(len(test)-1):
        value = test[i]
        count =0
        for row in rows:
            if row[i]==value:
                count = count+1
        attr_prob *= (count +0.1)/float((len(rows)+0.1)+(0.1*unique_f[i]))
        
        
    return attr_prob

def get_cond_probabilities(separated,test):
    summary = dict()
    for class_label,rows in separated.items():
        summary[class_label] = get_class_cond_probabilities(test,rows)
    
    return summary   

def cal_joint_probabilities(class_prob, class_cond_prob):
    joint_prob = dict()
    for class_value in class_prob:
        joint_prob[class_value]= class_prob[class_value]*class_cond_prob[class_value]
    
    return joint_prob

def naive_bayes(train, test):
    separated = separate_by_class(train)
    class_probabilities = get_class_probabilities(separated, train)
    predictions = []
    for row in test:
        class_cond_prob = get_cond_probabilities(separated, row)
        joint_probability = cal_joint_probabilities(class_probabilities, class_cond_prob)
        output = max(joint_probability, key=joint_probability.get)
        predictions.append(output)
    return predictions
    
train, test = get_data()
predictions =naive_bayes(train, test)

for i in predictions:
    print(i)

