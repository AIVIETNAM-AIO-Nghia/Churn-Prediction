import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


class ChurnPipeline:

    def clean(self, df):
        df = df.copy()

        #remove unneccessary columns
        df = df.drop(columns=['id', 'full_name', 'address'])

        #change date format
        df['created_date'] = pd.to_datetime(df['created_date'], dayfirst=True, errors='coerce')
        df['last_active_date'] = pd.to_datetime(df['last_active_date'], errors='coerce')

        #filter the incorrect data, month value should be <12
        df.loc[df['last_transaction_month'] > 12, 'last_transaction_month'] = np.nan

        #filter missing values of number and fill it with median value
        for col in df.select_dtypes(include='number'):
            df[col] = df[col].fillna(df[col].median())

        #filter missing values of categories type and replace it with Unknown
        for col in df.select_dtypes(include='object'):
            df[col] = df[col].fillna('Unknown')

        return df
