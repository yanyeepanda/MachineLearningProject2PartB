__author__ = 'Yanyi'

from glob import glob

allTrainFile = 'data/*.csv'
label_origin = []
instance_list = []

# get raw data for features from suburb profile files
for doc in glob(allTrainFile):
    feature_dict = {}
    with open(doc, 'rb') as file:
        for i, line in enumerate(file):
            # get top 3 industries
            if i == 158:
                feature_dict['top_ind'] = line.split(',')[2]
            if i == 160:
                feature_dict['2_ind'] = line.split(',')[2]
            if i == 162:
                feature_dict['3_ind'] = line.split(',')[2]
            # get top 3 occupations
            if i == 164:
                feature_dict['top_occup'] = line.split(',')[2]
            if i == 166:
                feature_dict['2_occup'] = line.split(',')[2]
            if i == 168:
                feature_dict['3_occup'] = line.split(',')[2]
    instance_list.append(feature_dict)

# turn categorical features to attribute=value as key, and the value is 0 or 1 to represent true or false
from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer()
instances_vectorized = v.fit_transform(instance_list)
allCatInstances = instances_vectorized.toarray().tolist()

# get raw data for label
for doc in glob(allTrainFile):
    with open(doc, 'rb') as file:
        for i, line in enumerate(file):
            if i == 126:
                low_income_household = float(line.split(',')[3])
    label_origin.append(low_income_household)

# put labels to 2 or 3 types
label = []
for p in label_origin:
    if p < 30:
        label.append(0)
    elif 30 <= p < 40:
        label.append(1)
    else:
        label.append(2)
    # if p < 30:
    #     label.append(0)
    # else:
    #     label.append(1)

# save some instances as test data
train_feature_data = allCatInstances[:30]
test_feature_data = allCatInstances[30:]

# save features to pickle file
import cPickle as pickle
train_file = open("train_feature.p", "wb")
pickle.dump(train_feature_data, train_file)
train_file.close()

label_file = open("label.p", "wb")
pickle.dump(label[:30], label_file)
label_file.close()

test_file = open("test_feature.p", "wb")
pickle.dump(test_feature_data, test_file)
train_file.close()

test_label_file = open("test_label.p", "wb")
pickle.dump(label[30:], test_label_file)
label_file.close()
