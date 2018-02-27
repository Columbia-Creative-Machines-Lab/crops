from shutil import copy2
import pickle

fps = pickle.load(open('false_positives.pkl', 'rb'))


for fp in fps:
  fp_name = fp.split("/")[-1].split(".")[0]
  copy2(fp, './fps/' + fp_name + "-fp.jpg")  
