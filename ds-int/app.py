import boto3
import awswrangler as wr
import typer,os,typing,time
import pandas as pd
import yaml

from core.date_utils import *
from core.db.db_utils import *


from datetime import datetime, date, timedelta
from scipy import stats

from dateutil.relativedelta import *
from dateutil.easter import *
from dateutil.rrule import *
from dateutil.parser import * 

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner




        
def print(message:str):
    """_summary_

    Args:
        message (str): prints message in prefect
    """
    logger = get_run_logger()
    logger.info(message) 
 
@task
def get_core_db_exports(glue_mysql_connection:str,delivered_at_begin:str,delivered_at_end:str) -> pd.DataFrame:
    """_summary_

    Args:
        glue_mysql_connection (str): _description_
        delivered_at_begin (str): _description_
        delivered_at_end (str): _description_

    Returns:
        pd.DataFrame: _description_
    """    
    sql = f"""
                select distinct 
                    ex.created_at,
                    ex.delivered_at,
                    te.name tenant_name,
                    ch.code channel_code,
                    ch.name channel_name,
                    case 
                        when ex.`type` = 'd' then "Delivery" 
                        when ex.`type` = 't' then "TakeDown" 
                        when ex.`type` = 'u' then "Update" 
                        else "n/a" end as export_type
                    ,ROUND((UNIX_TIMESTAMP(ex.delivered_at) - UNIX_TIMESTAMP(ex.created_at)) / 3600, 2) AS hours 
                FROM exports ex left join tenants te on te.id = ex.tenant_id 
                left join channels ch on ch.id = ex.channel_id 
                WHERE delivered_at >= '{delivered_at_begin} 00:00:00' AND delivered_at <= '{delivered_at_end} 23:59:59' 
                ORDER BY delivered_at  
                ;
        """
    # print(sql)
    print(f'Getting Exports from {delivered_at_begin} to {delivered_at_end} ...') 
    df = get_data(glue_mysql_connection,sql,"mysql")
    print('...done!')
    return df
  
cli = typer.Typer()

@flow(name="Dataproduct-Deliveries",task_runner=SequentialTaskRunner())
def main( 
    glue_mysql_connection: str = "mx2_prod_aurora"
    ,destination_bucket: str = "sonosuite-db-out"
    ,destination_prefix: str = "dataproducts/deliveries/exports"
    ,glue_crawler: str = "dataproducts-deliveries-exports" 
    ):
    """Extracts from core DB Deliveries for the specified dates

    Args:
        glue_mysql_connection (str, optional): _description_. Defaults to "mx2_prod_aurora".
        destination_bucket (str, optional): _description_. Defaults to "sonosuite-db-out".
        destination_prefix (str, optional): _description_. Defaults to "dataproducts/deliveries". 
    """

    #st.write(os.environ['AWS_DEFAULT_REGION'])
    print("Reading Configuration Pipeline... ") 
   

    #to get the current working directory
    APP_FOLDER = os.getcwd()
    with open(f'{APP_FOLDER}/cfg/app.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        CONFIG_APP = yaml.load(file, Loader=yaml.FullLoader)

    print(CONFIG_APP) 
 

if __name__ == '__main__': 
     
    typer.run(main)