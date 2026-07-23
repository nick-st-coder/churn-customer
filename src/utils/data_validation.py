import great_expectations as gx
from typing import List, Tuple

def data_validation(df) -> Tuple[bool, List[str]]:
    gx_df = gx.from_pandas(df)

    gx_df

def test():
    import great_expectations as gx

    return(gx.__version__)    
