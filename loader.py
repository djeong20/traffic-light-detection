import os
import numpy as np
import pandas as pd

# from sklearn.model_selection import train_test_split

class LISA():
    # Load CSV file to DataFrame
    def load_annotations(self, path):
        dir = sorted(os.listdir(path))
        df_list = []

        for name in dir:
            if 'dayClip' in name or 'nightClip' in name:
                df_list.append(pd.read_csv(os.path.join(path, name, 'frameAnnotationsBOX.csv'), sep=';'))
            
        df = pd.concat(df_list, axis=0)

        return df

    def data_cleaning(self, df):
        # 1. Drop duplicate columns
        df = df.drop(['Origin file', 'Origin track', 'Origin track frame number'], axis=1)

        # 2. Change column name for convenience
        df.columns = ['image_id', 'label', 'x_min', 'y_min', 'x_max', 'y_max', 'frame', 'night']

        # 3. Change image id to corresponding path to data
        def changeFilename(x):
            filename = x.image_id
            splitted = filename.split('/')
            name = splitted[-1].split('--')[0]

            if x.night:
                return os.path.join('dataset', f'nightTrain/nightTrain/{name}/frames/{splitted[-1]}')
            else:
                return os.path.join('dataset', f'dayTrain/dayTrain/{name}/frames/{splitted[-1]}')

        df['image_id'] = df.apply(changeFilename, axis=1)

        # 4. Change label go, warning, stop to int 1, 2, 3
        def changeLabel(x):
            tag = x.label

            if tag == 'go': return 1
            elif tag == 'warning': return 2
            else: return 3

        df['label'] = df.apply(changeLabel, axis=1)

        return df
    
    def load_dataset(self, day_path, night_path):
        df_day = self.load_annotations(day_path)
        df_day['night'] = 0

        df_night = self.load_annotations(night_path)
        df_night['night'] = 1

        df = self.data_cleaning(pd.concat([df_day, df_night], axis=0))

        return df

    def train_test_split(self, df, p=0.25): # proportion of test dataset 
        df['name'] = df[['image_id']].applymap(lambda x: x.split('/')[3])
        
        days = ['dayClip1', 'dayClip2', 'dayClip3', 'dayClip4', 'dayClip5', 'dayClip6', 'dayClip7', 'dayClip8', 'dayClip9', 'dayClip10', 'dayClip11', 'dayClip12', 'dayClip13']
        day_test = list(np.random.choice(days, int(p * 13)))
        day_train = list(set(days) - set(day_test))

        nights = ['nightClip1', 'nightClip2', 'nightClip3', 'nightClip4', 'nightClip5']
        night_test = list(np.random.choice(nights, int(p * 5)))
        night_train = list(set(nights) - set(night_test))
        
        train = night_train + day_train
        df_train = df[df.name.isin(train)]

        test = night_test + day_test
        df_test = df[df.name.isin(test)]
        
        return df_train, df_test