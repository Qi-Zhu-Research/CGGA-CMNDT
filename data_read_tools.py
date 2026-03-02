import os
import pandas as pd
import pickle


def save_to_pickle(obj, filepath):
    # Ensure the file's directory exists
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check the type of the object and save accordingly
    if isinstance(obj, pd.DataFrame):
        obj.to_pickle(filepath)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)


def load_from_pickle(filepath):
    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"No file found at {filepath}")
        return None

    # Load object from pickle
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    return obj

class DiffTool:
    @staticmethod
    def compare_and_diff_dataframes(df1, df2, df1_name, df2_name):
        if not df1.equals(df2):
            comparison_df = df1.eq(df2)
            rows_with_difference = comparison_df[comparison_df.all(axis=1) == False]
            if not rows_with_difference.empty:
                print(f"Differences between {df1_name} and {df2_name}:")
                print(rows_with_difference)
            else:
                print(f"No differences between {df1_name} and {df2_name}")
        else:
            print(f"No differences between {df1_name} and {df2_name}")

    @staticmethod
    def compare_and_diff_dicts(dict1, dict2, dict1_name, dict2_name):
        diff_keys = [k for k in dict1 if dict1.get(k) != dict2.get(k)]
        if diff_keys:
            print(f"Differences between {dict1_name} and {dict2_name}:")
            for key in diff_keys:
                print(f"Key: {key}, {dict1_name}: {dict1.get(key)}, {dict2_name}: {dict2.get(key)}")
        else:
            print(f"No difference between {dict1_name} and {dict2_name}")