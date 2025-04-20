# This code is useful to clean/process/prepare the data to be used for ML/AI pipelines.

import pandas as pd
import numpy as np

def check_drop_duplicates(df, subset_cols, drop=True):
    """
    Check for and optionally drop duplicates in a DataFrame based on a list of column names.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    subset_cols : list of str
        Column names to check for duplicates.
    drop : bool, default=True
        Whether to drop duplicates. If True, keeps the first occurrence.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with duplicates optionally removed.
    """
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Sanity check
    if not all(col in df.columns for col in subset_cols):
        raise ValueError("Some columns in 'subset_cols' are not in the DataFrame.")

    # Ensure datetime conversion for datetime-like columns (optional, only if needed)
    for col in subset_cols:
        if 'date' in col.lower() or 'time' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='ignore')

    # Find duplicates
    duplicates = df[df.duplicated(subset=subset_cols, keep=False)]

    if not duplicates.empty:
        print(f"Found {len(duplicates)} duplicate rows based on {subset_cols}:\n")
        print(duplicates.sort_values(by=subset_cols))

        if drop:
            df = df.drop_duplicates(subset=subset_cols, keep='first')
            print(f"\nDuplicates removed. New shape: {df.shape}")
        else:
            print("\nNo duplicates removed.")
    else:
        print(f"No duplicate values found based on {subset_cols}.")

    return df


def fix_quarterly_kwh_groups(df, time_col='start_date_UTC', value_col='kwh'):
    """
    Detects and corrects incorrect [0, 0, 0, total] energy patterns in quarter-hour data
    by replacing them with [total/4]*4 when the four timestamps belong to the same hour.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least [time_col, value_col]
    time_col : str
        Name of datetime column with quarter-hour timestamps
    value_col : str
        Name of the column with kWh values

    Returns
    -------
    df_fixed : pd.DataFrame
        Copy of df with fixed 'kwh' groups
    modified_indices : list
        List of row indices that were modified
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    # Sort by timestamp
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # Extract hour, minute, day
    hours = df[time_col].dt.hour.to_numpy()
    minutes = df[time_col].dt.minute.to_numpy()
    days = df[time_col].dt.date.to_numpy()
    kwh_values = df[value_col].to_numpy()

    modified_indices = []

    # Loop through in steps of 1 to find valid 4-quarter groups
    for i in range(0, len(df), 4):
        h_group = hours[i:i+4]
        m_group = minutes[i:i+4]
        d_group = days[i:i+4]

        # Check if all 4 are in the same hour and on same day and minutes = [0,15,30,45]
        if (
            len(h_group) == 4 and
            np.all(h_group == h_group[0]) and
            np.all(d_group == d_group[0]) and
            np.array_equal(m_group, [0, 15, 30, 45])
        ):
            group_kwh = kwh_values[i:i+4]
            if np.array_equal(group_kwh[1:], [0, 0, 0]) and group_kwh[0] > 0:
                # Fix it: evenly distribute total to all 4
                new_val = group_kwh[0] / 4
                kwh_values[i:i+4] = [new_val] * 4
                modified_indices.extend(range(i, i+4))

    # Assign fixed values back
    df[value_col] = kwh_values
    return df, modified_indices


def check_time_spacing(df, time_col='start_date_UTC', expected_freq='15min', verbose=True):
    """
    Checks if time intervals in a DataFrame follow the expected frequency.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing timestamps.
    time_col : str
        The column name containing datetime values.
    expected_freq : str
        Expected frequency between rows (e.g., '15min', '1H', '1D', '1M').
    verbose : bool
        Whether to print detailed info. Set to False if using in scripts.

    Returns:
    --------
    pd.DataFrame :
        A DataFrame showing where the gaps are (if any), with previous and current timestamps.
    """

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    expected_delta = pd.to_timedelta(pd.tseries.frequencies.to_offset(expected_freq))
    df['prev_time'] = df[time_col].shift(1)
    df['actual_delta'] = df[time_col] - df['prev_time']

    # Find where actual delta != expected
    gaps = df[df['actual_delta'] != expected_delta].copy()

    if verbose:
        if gaps.empty:
            print(f"All time intervals in '{time_col}' match the expected frequency of {expected_freq}.")
        else:
            print(f"Found {len(gaps)} gaps that do NOT match expected frequency of {expected_freq}:")
            print(gaps[[time_col, 'prev_time', 'actual_delta']])

    return gaps[[time_col, 'prev_time', 'actual_delta']]
