import boto3,time
from date_utils import print_timestamp


'''
AWS related functions and tasks
'''

def run_crawler(glue_crawler:str,region_name,refresh_seconds):
    """runs AWS Glue Crawler to update Glue Catalog. So data in new partitions is visible in Athena / Redshift Spectrum

    Args:
        name (str): Crawler Name
    """

    glue = boto3.client("glue", region_name=region_name)
    if get_crawler(glue, glue_crawler)['Crawler']['State'] not in ['STARTING','RUNNING','STOPPING']:
        response =  glue.start_crawler(  Name=glue_crawler ) 
        print( f"{print_timestamp()}:{response}")
    while get_crawler(glue, glue_crawler)['Crawler']['State']  in ['STARTING','RUNNING']: 
        print( f"{print_timestamp()}:Crawler {get_crawler(glue, glue_crawler)['Crawler']['State']}" )
        time.sleep(refresh_seconds)

def get_crawler(glue, glue_crawler):
    response =  glue.get_crawler(Name=glue_crawler)
    return response