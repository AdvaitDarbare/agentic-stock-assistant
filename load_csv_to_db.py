import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 1️⃣ Load environment variables
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# 2️⃣ Build DB URI
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 3️⃣ Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# 4️⃣ Create table if it doesn't exist
create_table_sql = """
CREATE TABLE IF NOT EXISTS stock_data (
    ticker TEXT,
    date DATE,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT
)
"""

with engine.connect() as conn:
    conn.execute(text(create_table_sql))
    conn.commit()
    print("✅ Table created (or already exists)")

# 5️⃣ Load and insert all CSVs
data_folder = "NASDAQ100"
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

for csv_file in csv_files:
    path = os.path.join(data_folder, csv_file)
    ticker = csv_file.replace('.csv', '')

    print(f"📥 Loading {csv_file}")
    df = pd.read_csv(path)

    # Add ticker column
    df['ticker'] = ticker

    # Ensure columns are correct order
    df = df[['ticker', 'date', 'open', 'high', 'low', 'close']]

    # Insert
    df.to_sql('stock_data', engine, if_exists='append', index=False)
    print(f"✅ Inserted {len(df)} rows from {csv_file}")

print("🎉 All CSV files loaded successfully!")
