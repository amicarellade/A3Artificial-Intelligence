import pandas as pd

def preprocess_data(file):
    """
    Preprocesses the data by converting object columns into indexes and labeling them as categorical.

    Parameters:
        file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    df = pd.read_csv(file)
    df = df.drop(df.columns[0], axis=1)

    category_columns = [column for column in df.columns if df[column].dtype == 'object']

    mapping_functions = {}
    for column in category_columns:
        values = df[column].unique()
        mapping_function = {value: idx for idx, value in enumerate(values)}
        mapping_functions[column] = mapping_function

    for column in category_columns:
        df[column] = df[column].map(mapping_functions[column])

    for column in category_columns:
        df[column] = df[column].astype('category')

    return df
