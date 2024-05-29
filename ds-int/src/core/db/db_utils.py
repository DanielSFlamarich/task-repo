import awswrangler as wr
import typing
import pandas as pd


def get_sql_query(path: str, option: str = 'r'):
    '''
    Get sql queries from a file using a context manager

    Args:
        path (str): Path where the query is
        option (str): Read as default 'r'

    Returns:
        SQL text query
    '''

    with open(path, option) as q:
        query = q.read()
        return query

# this function IN
def get_data(glue_connection, sql:str, connection_type:str="redshift") :
    """_summary_

    Args:
        glue_connection (_type_): glue Catalog connection name
        sql (str): sql statement to execute
        connection_type (str, optional): Connection type: "mysql" or "redshift". Defaults to "redshift".

    Returns:
        Union[None,pd.DataFrame]: Pandas Dataframe with results or None when no results
    """    

    if connection_type == "redshift":
        with wr.redshift.connect(glue_connection) as con_db:
            df = wr.redshift.read_sql_query(sql, con=con_db)
            return df
    elif connection_type == "mysql":
        with wr.mysql.connect(glue_connection) as con_db:
            df = wr.mysql.read_sql_query(sql, con=con_db)
            return df

def execute_multiple_raw_sqls(  sql_stmts:typing.List[str]
                                ,glue_db_connection:str
                                ,connection_type:str):
    """
    Execute Raw SQL queries (No Pandas)
    
    Args:
        sql_stmts (typing.List[str]): List of Sql queries
        glue_connection (str): glue catalog database connection name
        wr: AWS Datawrangler object 
    """
    #print(sql)
    if connection_type == "redshift":
        with wr.redshift.connect(glue_db_connection) as con_db:
            with con_db.cursor() as cursor:
                for stmt in sql_stmts:
                    cursor.execute(stmt)
    elif connection_type == "mysql":
        with wr.mysql.connect(glue_db_connection) as con_db:
            with con_db.cursor() as cursor:
                for stmt in sql_stmts:
                    cursor.execute(stmt)

                
                
def print_multiple_raw_sqls(sql_stmts:list):
    """prints the each of the sqls in the sql_stmts list

    Args:
        sql_stmts (list): list of sql statements
    """
    for stm in sql_stmts:
        print(stm)

def save_dataframe_to_s3_parquet(df: pd.DataFrame,destination_file_path):

    wr.s3.to_parquet(df,path=destination_file_path) 