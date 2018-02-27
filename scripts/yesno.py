import os 
import pickle
import csv
import glob
from tqdm import tqdm
from random import shuffle

source = os.path.dirname(__file__)
parent = os.path.join(source, '../')
csvs = glob.glob(os.path.join(parent, "data/Count/*.csv"))
non_lesions=[]
lesions=[]
 
for c in tqdm(csvs):
    csfile = open(c, 'r')
    csfreader = csv.reader(csfile)
    line_list = []
    rows = []

    for row in csfreader:
        rows.append(row)

    file_name = rows[1][1] + ".jpg"

    if len(rows) > 2:
        lesions.append(file_name)
        continue

    l = rows[1] 
    this_line = ((float(l[2]),float(l[3])), (float(l[4]),float(l[5])))
    if this_line != ((0,0), (0,0)):
        lesions.append(file_name)
        continue
    non_lesions.append(file_name)

shuffle(lesions)
shuffle(non_lesions)

yes_len = len(lesions)
no_len = len(non_lesions)

yes_train = lesions[:int(yes_len * .7)]
yes_val = lesions[int(yes_len * .7):int(yes_len * .85)]
yes_test = lesions[int(yes_len * .85):]

no_train = non_lesions[:int(no_len * .7)]
no_val = non_lesions[int(no_len * .7):int(no_len * .85)]
no_test = non_lesions[int(no_len * .85):]

pickle.dump(yes_train, open("yes_train.pkl", 'wb'))
pickle.dump(yes_val, open("yes_val.pkl", 'wb'))
pickle.dump(yes_test, open("yes_test.pkl", 'wb'))
pickle.dump(no_train, open("no_train.pkl", 'wb'))
pickle.dump(no_val, open("no_val.pkl", 'wb'))
pickle.dump(no_test, open("no_test.pkl", 'wb'))
