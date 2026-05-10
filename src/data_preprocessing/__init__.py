import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ChurnPipeline:

    def clean(self, df):
        df = df.copy()

        df = df.drop(columns=['id', 'full_name', 'address'])

        #Convert text to datetime format
        df['created_date'] = pd.to_datetime(df['created_date'], dayfirst=True, errors='coerce')
        df['last_active_date'] = pd.to_datetime(df['last_active_date'], errors='coerce')

       # Convert transaction/activity metric to numeric
        df['last_transaction_month'] = pd.to_numeric(
            df['last_transaction_month'],
            errors='coerce'
        )
        #Find and replace missing value with median value
        for col in df.select_dtypes(include='number'):
            df[col] = df[col].fillna(df[col].median())

        for col in df.select_dtypes(include='object'):
            df[col] = df[col].fillna('Unknown')

        return df

    def feature_engineering(self, df):
        df = df.copy()
        #Set a fixed reference date
        today = pd.Timestamp("2026-01-01")

        # Fill missing dates
        df['last_active_date'] = df['last_active_date'].fillna(df['created_date'])

        # Date features
        df['recency_days'] = (today - df['last_active_date']).dt.days

        df['tenure_days'] = (today - df['created_date']).dt.days

        # Prevent weird ratios
        df['monthly_ir'] = df['monthly_ir'].clip(lower=0)

        # Log transforms
        df['log_balance'] = np.log1p(df['balance'])

        df['log_income'] = np.log1p(df['monthly_ir'])

        df['log_last_transaction'] = np.log1p(df['last_transaction_month'])

        # Ratios
        df['balance_income_ratio'] = (df['balance'] / (df['monthly_ir'] + 1))

        df['transaction_income_ratio'] = (df['last_transaction_month'] /(df['monthly_ir'] + 1))

        # Drop unused columns
        df = df.drop(columns=[
            'risk_score',
            'cluster_group',
            'created_date',
            'last_active_date'
        ], errors='ignore')

        return df

    
    def transform(self, df):
        df = df.copy()
        #Get all numeric columns
        num_cols = df.select_dtypes(include=np.number).columns
        #Get all categorical columns
        cat_cols = df.select_dtypes(include='object').columns

        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        if len(cat_cols) > 0:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded = encoder.fit_transform(df[cat_cols])

            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(cat_cols)
            )

            df = df.drop(columns=cat_cols).reset_index(drop=True)
            df = pd.concat([df, encoded_df], axis=1)

        return df
    # =========================
    # FULL PIPELINE
    # =========================
    def process(self, df):
        df = self.clean(df)
        df = self.feature_engineering(df)
        df = self.transform(df)

        return df

# Read raw CSV file
raw_df = pd.read_csv("/kaggle/input/datasets/thuandao/bank-customer-behavior-and-churn-dataset/bank_churn_dataset.csv")

# PROCESSING SECTION
pipeline = ChurnPipeline()

processed_df = pipeline.process(raw_df)

# Save processed data to new CSV
processed_df.to_csv(
    "processed_churn_data.csv",
    index=False
)
print("\nPipeline completed successfully.")
