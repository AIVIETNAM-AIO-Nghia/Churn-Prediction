import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


class ChurnPipeline:

    def clean(self, df):
        df = df.copy()

        df = df.drop(columns=['id', 'full_name', 'address'])

        df['created_date'] = pd.to_datetime(df['created_date'], dayfirst=True, errors='coerce')
        df['last_active_date'] = pd.to_datetime(df['last_active_date'], errors='coerce')

        df.loc[df['last_transaction_month'] > 12, 'last_transaction_month'] = np.nan

        for col in df.select_dtypes(include='number'):
            df[col] = df[col].fillna(df[col].median())

        for col in df.select_dtypes(include='object'):
            df[col] = df[col].fillna('Unknown')

        return df
