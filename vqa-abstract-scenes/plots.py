import pickle

with open('third_model_hist', 'rb') as handle:
    b = pickle.load(handle)

print(b)