from LambdaData import LambdaData
import pickle

with open('100-a.pckl', 'rb') as f:
    lambdas, trace = pickle.load(f)
    print(trace)
