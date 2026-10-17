from utils.logger import get_logger
from sqlalchemy import create_engine
import pandas as pd

logger = get_logger(__name__)

def read_dataset(path: str) -> pd.DataFrame:
    logger.info('read dataset')
    engine = create_engine(f'sqlite:///{path}')
    df = pd.read_sql_query(
        "SELECT timestamp, heat_efficiency FROM heat_exchanger ORDER BY timestamp",
        engine,
    )
    df['timestamp'] = pd.to_datetime(df["timestamp"])
    df['day_index'] = (df['timestamp'] - df['timestamp'].min()).dt.days 
    return df


print(read_dataset('data/heat_exchanger.db'))