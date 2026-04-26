import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

with open('params.yaml') as f:
    params = yaml.safe_load(f)

df = pd.read_csv('data/raw/data.csv')
train, test = train_test_split(df, test_size=params['split_ratio'], random_state=params['random_state'])
train.to_csv('data/processed/train.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)
print('Done: train', len(train), 'test', len(test))
