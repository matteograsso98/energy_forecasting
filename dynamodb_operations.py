import boto3
import time
import pandas as pd
import numpy as np
from boto3.dynamodb.conditions import Attr
from boto3.dynamodb.conditions import Key
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
from google_auth_oauthlib.flow import InstalledAppFlow
#from googleapiclient.discovery import build
#from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
import os
import json

# Google Drive API scope
SCOPES = ['https://www.googleapis.com/auth/drive.file']


def authenticate_drive():
     """
     Google Drive authentication.
     We need Google Drive access to store heavy datasets
     """
     creds = None
     if os.path.exists('token.json'):
         creds = Credentials.from_authorized_user_file('token.json', SCOPES)
     else:
         flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
         creds = flow.run_local_server(port=0)
         with open('token.json', 'w') as token:
             token.write(creds.to_json())

     return build('drive', 'v3', credentials=creds)


def upload_file_to_drive(service, file_path, mime_type='text/csv'):
     """
     Upload the data set(s) to Google Drive.
     """
     file_metadata = {'name': os.path.basename(file_path)}
     media = MediaFileUpload(file_path, mimetype=mime_type)
     uploaded = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
     print(f"Uploaded to Google Drive with file ID: {uploaded['id']}")


dynamodb = boto3.resource('dynamodb', region_name='eu-central-1')  # Initialize the DynamoDB


def get_table_elements_fast(table_name, plant_prefix, local_output_path="_data_prod.csv"):
    """
    Query ALL the elements from a dynamoDB based on 'upEnd_Date_UTCIndex',
    the plant name AND the 'end_date_UTC' BEFORE 2025/12 (to take all the elements in the table)
    :param table_name (str): name of the dynamoDB table
    :param plant_prefix (str): name of the plant
    :local_output_path (str): name of the csv file to save
    """
    all_items = []
    dynamodb = boto3.resource('dynamodb', region_name='eu-central-1')
    table = dynamodb.Table(table_name)

    start_time = time.time()
    last_evaluated_key = None

    while True:
        query_kwargs = {
            'IndexName': 'upEnd_Date_UTCIndex',
            'KeyConditionExpression': Key('up').eq(plant_prefix) & Key('end_date_UTC').lt("2025/12"),
        }

        if last_evaluated_key:
            query_kwargs['ExclusiveStartKey'] = last_evaluated_key

        response = table.query(**query_kwargs)
        items = response['Items']
        all_items.extend(items)

        last_evaluated_key = response.get('LastEvaluatedKey')
        if not last_evaluated_key:
            break

    print(f"Queried {len(all_items)} items in {time.time() - start_time:.2f} seconds.")

    # Convert Decimals and build DataFrame
    def clean_item(item):
        return {k: float(v) if isinstance(v, Decimal) else v for k, v in item.items()}

    df = pd.DataFrame([clean_item(item) for item in all_items])

    df = df.sort_values(by='start_date_UTC')

    # Save locally
    df.to_csv(plant_prefix + local_output_path, index=False)
    print(f"CSV saved locally to: {local_output_path}")
    return local_output_path


def get_table_elements(table_name, plant_prefix):
    """
    Get the elements of a dynamodb table starting with 'plant_prefix'
    ################################################################
    #################### THIS IS SLOW ##############################
    ################################################################
    :param table_name:  the table name (string)
    :param plant_prefix: the prefix of the photovoltaic or wind plant. E.g., WN_MONTECALVELLO
    """
    print('this is fucking slow. Use get_table_elements_fast (if possible) instead!')
    all_items = []  # For collecting the results
    start_time = time.time()
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    # plant_prefix = "Kilbraur"
    count = 0
    last_evaluated_key = None

    while True:
        if last_evaluated_key:
            response = table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('hashKey').begins_with(plant_prefix),
                ExclusiveStartKey=last_evaluated_key
            )
        else:
            response = table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('hashKey').begins_with(plant_prefix)
            )

        items = response['Items']
        all_items.extend(items)  # Save to list
        count += len(items)

        last_evaluated_key = response.get('LastEvaluatedKey')
        if not last_evaluated_key:
            break

    print(f"Number of items with hash_key starting with '{plant_prefix}': {count}")
    print("--- %s seconds ---" % (time.time() - start_time))

    # Optional: convert Decimal to float (DynamoDB stores numbers as Decimal)
    def clean_item(item):
        clean = {}
        for k, v in item.items():
            if isinstance(v, Decimal):
                clean[k] = float(v)
            else:
                clean[k] = v
        return clean

    # Create DataFrame
    cleaned_items = [clean_item(i) for i in all_items]
    df = pd.DataFrame(cleaned_items)

    # Display the first few rows
    print("\nData preview:")
    print(df.head())

    # Optional: sort by timestamp (if column exists)
    if 'sortKey' in df.columns:
        df['sortKey'] = pd.to_datetime(df['sortKey'], errors='coerce')
        df = df.sort_values(by='sortKey')

    # Save to CSV or Excel if you like
    return df.to_csv(plant_prefix + "_data.csv", index=False)


def update_item_in_dynamodb_test(hash_key_value, sort_key_value, updated_values):
    """
    Simulate updating an item in DynamoDB by printing the update request.

    :param hash_key_value: The value of the hashKey (Partition Key).
    :param sort_key_value: The value of the sortKey (Sort Key).
    :param updated_values: A dictionary containing the attribute names and their new values.
    """

    # Simulate the key and update expression
    key = {
        'hashKey': hash_key_value,
        'sortKey': sort_key_value
    }

    update_expression = "SET " + ", ".join([f"{k} = :{k}" for k in updated_values.keys()])
    expression_attribute_values = {f":{k}": v for k, v in updated_values.items()}

    print(f"[SIMULATED UPDATE] Would update item with key: {key}")
    print(f"UpdateExpression: {update_expression}")
    print(f"ExpressionAttributeValues: {expression_attribute_values}")
    print("-" * 60)


def update_item_in_dynamodb(hash_key_value, sort_key_value, updated_values,table_name):
    """
    Update an item in DynamoDB with given hashKey, sortKey, and updated values.

    :param hash_key_value: The value of the hashKey (Partition Key).
    :param sort_key_value: The value of the sortKey (Sort Key).
    :param updated_values: A dictionary containing the attribute names and their new values.
    :param table (string): Name of the dynamodb table I am going to update
    """
    dynamodb = boto3.resource('dynamodb', region_name='eu-central-1')  # Initialize the DynamoDB
    table = dynamodb.Table(table_name)

    try:
        # Build the key for identifying the item to update
        key = {
            'hashKey': hash_key_value,  # Partition Key (Hash Key)
            'sortKey': sort_key_value  # Sort Key (Range Key)
        }

        # Build the update expression and attribute values
        update_expression = "SET " + ", ".join([f"{k} = :{k}" for k in updated_values.keys()])
        expression_attribute_values = {f":{k}": v for k, v in updated_values.items()}

        # Perform the update
        response = table.update_item(
            Key=key,
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_attribute_values,
            ReturnValues="UPDATED_NEW"  # Return the updated values after the update
        )

        # Print the response or handle as needed
        print(f"Item updated successfully: {response}")

    except Exception as e:
        print(f"Error updating item: {e}")


"""
# Example usage
hash_key_value = 'WN_MONTECALVELLO#PlantMeterEnergyWind#2023-12-31T23:45:00.000Z'  # Replace with the actual hash key value
sort_key_value = '2024-09-18T12:14:42.721Z'  # Replace with the actual sort key value
updated_values = {
    'kwh': Decimal(str(0.001)),
}

update_item_in_dynamodb(hash_key_value, sort_key_value, updated_values)
"""


def batch_update_from_dataframe(table_name, df):
    """
    Loop through the dataframe and update each item in DynamoDB.
    Assumes the dataframe has 'hashKey', 'sortKey', and 'kwh' columns.
    """
    for idx, row in df.iterrows():
        hash_key = row['hashKey']
        sort_key = row['sortKey']
        kwh_value = Decimal(str(row['kwh']))

        updated_values = {'kwh': kwh_value}
        update_item_in_dynamodb(hash_key, sort_key, updated_values,table_name)


def update_item_in_dynamodb(hash_key, sort_key, updated_values, table_name):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    response = table.update_item(
        Key={
            'hashKey': hash_key,
            'sortKey': sort_key
        },
        UpdateExpression='SET kwh = :kwh',
        ExpressionAttributeValues={
            ':kwh': updated_values['kwh']
        },
        ReturnValues='UPDATED_NEW'
    )
    return response

def parallel_update(table_name, df, max_workers=10,
                    checkpoint_file='updated_items.json',
                    failed_file='failed_items.json'):
    start_time = time.time()
    updated_items = set()
    failed_items = set()

    # Load previously updated items
    try:
        with open(checkpoint_file, 'r') as f:
            updated_items = set(json.load(f))
    except FileNotFoundError:
        pass

    # Load previously failed items (optional)
    try:
        with open(failed_file, 'r') as f:
            failed_items = set(json.load(f))
    except FileNotFoundError:
        pass

    futures = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _, row in df.iterrows():
                key = f"{row['hashKey']}|{row['sortKey']}"
                if key in updated_items:
                    continue  # Already done

                hash_key_value = row['hashKey']
                sort_key_value = row['sortKey']
                updated_values = {
                    'kwh': Decimal(str(row['kwh']))
                }

                future = executor.submit(
                    update_item_in_dynamodb,
                    hash_key_value,
                    sort_key_value,
                    updated_values,
                    table_name
                )
                future.key = key
                futures.append(future)

            for future in as_completed(futures):
                key = future.key
                try:
                    future.result()
                    updated_items.add(key)
                except Exception as e:
                    print(f"Failed to update item {key}: {e}")
                    failed_items.add(key)

    except KeyboardInterrupt:
        print("\nRun interrupted by user. Saving progress...")

    # Save progress
    with open(checkpoint_file, 'w') as f:
        json.dump(list(updated_items), f)

    with open(failed_file, 'w') as f:
        json.dump(list(failed_items), f)

    # Summary
    print("\n--- SUMMARY ---")
    print(f"Successfully updated: {len(updated_items)} items")
    print(f"Failed updates: {len(failed_items)} items")
    print(f"Total time: {time.time() - start_time:.2f} seconds")


def parallel_check(df, max_workers=20):
    """
    check for updated elements in the dynamodb
    Purpose: Before updating, use this function to:
    - See what’s already correct
    - Create a list of missing or incorrect items
    - You can use the result to filter df, and only send what still needs updating into parallel_update().
    After updating: use this function to:
    - Check if all the items have been updated

    :param df (pandas): data frame containing the updated values 
    :param max_workers: how many checks at  a time you want to perform. 
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(check_item, row)
                   for _, row in df.iterrows()
                   if row['up'] in target_plants]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    return results

    return df
