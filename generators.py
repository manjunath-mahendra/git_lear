import numpy as np
from library.generators.NextConvGeN import NextConvGeN
import pandas as pd
from fdc.fdc import feature_clustering, canberra_modified, Clustering, FDC
from sdv.single_table  import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model.ctabgan import CTABGAN
import json
import os

# added new line
def balanced_syn_data(syn_data, value_count, label):
    """
    This function balance the target of synthetic data as that of real data
    
    Args:
        syn_data (DataFrame): Synthetic data from the generative model
        value_count (Dictionary): Target value counts of real data
        label (string): Name of the target column
        
    Returns:
        balanced_sythetic_data (DataFrame): Balanced synthetic data
    
    """
    #columns = syn_data.columns
    df_list = []
    for class_label in value_count:
        if isinstance(class_label, str):
            class_label = float(class_label)
        if isinstance(class_label, float):
            class_label = int(class_label)
            
        class_df = syn_data[syn_data[label] == class_label].sample(n=value_count[str(class_label)], axis=0, random_state=42)
        df_list.append(class_df)
    balanced_synthetic_data = pd.concat(df_list)
    return balanced_synthetic_data.sample(frac=1, random_state=42)

def meta_data(data, categorical_features, numerical_features):
    column_dict = {}

    for column in data.columns:
        column_dict[column] = {}

        if column in categorical_features:
            column_dict[column]["sdtype"] = "categorical"
        elif column in numerical_features:
            column_dict[column]["sdtype"] = "numerical"
            column_dict[column]["computer_representation"] = "Float"
        else:
            column_dict[column]["sdtype"] = "unknown"

    return {"columns": column_dict}




def CTABGAN_pipeline(task = "semi-supervised", base_directory, synth_directory, balanced_synth_directory):
    # List all subdirectories (folders) in the base directory
    all_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

    # Iterate through all the folders
    for folder in all_folders:
        folder_path = os.path.join(base_directory, folder)

        # Check if the 'semi-supervised' directory exists in the current folder
        task_directory = os.path.join(folder_path, task)
        if os.path.exists(task_directory):

            # Check if 'training_data.csv' exists in the task directory
            training_data_path = os.path.join(task_directory, "training_data.csv")
            if os.path.exists(training_data_path):

                # Load the training data
                data = pd.read_csv(training_data_path)

                # Load additional info from the 'additional_info.json' file in the 'semi-supervised' directory
                info_path = os.path.join(task_directory, "additional_info.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r') as info_file:
                        info = json.load(info_file)

                    # Extract information from the info dictionary
                    categorical_columns = []
                    integer_columns = []

                    if info['indices_ordinal_features'] is not None:
                        categorical_columns.extend([info['ordered_features'][i] for i in info['indices_ordinal_features']])

                    if info['indices_nominal_features'] is not None:
                        categorical_columns.extend([info['ordered_features'][i] for i in info['indices_nominal_features']])

                    if info['indices_continuous_features'] is not None:
                        integer_columns.extend([info['ordered_features'][i] for i in info['indices_continuous_features']])
                    
                    if isinstance(info.get("target"), str):
                        target=info['target']
                    else:
                        target=info['target'][0]
                    
                    # Add the target column
                    if info['target'] is not None:
                        categorical_columns.append(target)


                    # Create the synthesizer with the extracted information
                    synthesizer = CTABGAN(raw_csv_path=training_data_path,
                                          categorical_columns=categorical_columns,
                                          integer_columns=integer_columns,
                                          problem_type={"Classification": target})

                    # Fit the synthesizer
                    synthesizer.fit()

                    # Generate synthetic samples
                    syn = synthesizer.generate_samples()
                    
                    syn[categorical_columns]=syn[categorical_columns].astype(float).astype(int)

                    # Balance the synthetic data
                    balanced_synthetic_data = balanced_syn_data(syn, info['target_value_counts'], target)

                    # Create path
                    syn_save_path = os.path.join(synth_directory,'CTABGAN', folder, task, "synthetic_data.csv")
                    balanced_synthetic_data_save_path = os.path.join(balanced_synth_directory, 'CTABGAN', folder, task, "synthetic_data.csv")

                    # Create the directories if they don't exist
                    os.makedirs(os.path.dirname(syn_save_path), exist_ok=True)
                    os.makedirs(os.path.dirname(balanced_synthetic_data_save_path), exist_ok=True)

                    syn.to_csv(syn_save_path, index=False)
                    balanced_synthetic_data.to_csv(balanced_synthetic_data_save_path, index=False)
                    
                    print("Generated Synthetic data for {} and stored in path sucessfully".format(folder))  

                else:
                    print(f"'additional_info.json' file not found in '{task_directory}'")

            else:
                print(f"'training_data.csv' not found in '{task_directory}'")

        else:
            print(f"'semi-supervised' directory not found in '{folder}'")
            


def NextConvGeN_pipeline(task = "semi-supervised", base_directory, synth_directory, balanced_synth_directory):
    # List all subdirectories (folders) in the base directory
    all_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]
    # Iterate through all the folders
    for folder in all_folders:
    #if folder not in completed:
        folder_path = os.path.join(base_directory, folder)

        # Check if the 'semi-supervised' directory exists in the current folder
        task_directory = os.path.join(folder_path, task)
        if os.path.exists(task_directory):

            # Check if 'training_data.csv' exists in the 'semi-supervised' directory
            training_data_path = os.path.join(task_directory, "NextConvGeN_training_data.csv")
            if os.path.exists(training_data_path):

                # Load the training data
                data = pd.read_csv(training_data_path)

                # Load additional info from the 'additional_info.json' file in the 'semi-supervised' directory
                info_path = os.path.join(task_directory, "additional_info.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r') as info_file:
                        info = json.load(info_file)

                    # Extract information from the info dictionary
                    ordered_features = info['ordered_features']

                    if isinstance(info.get("target"), str):
                        target=info['target']
                    else:
                        target=info['target'][0]

                    # Add the target column
                    if info['target'] is not None:
                        ordered_features.append(target)

                    n_syn_samples=data.shape[0]*5

                    fdc = FDC()
                    fdc.ord_list=info['indices_ordinal_features']
                    if info['indices_nominal_features'] is not None:
                        nominal_indices = info['indices_nominal_features']
                    else:
                        nominal_indices = []

                    fdc.nom_list = nominal_indices + [ordered_features.index(target)]

                    fdc.cont_list =info['indices_continuous_features']

                    train_features=np.array(data)


                    # Train the synthesizer 
                    gen = NextConvGeN(train_features.shape[1], neb=5, fdc=fdc,alpha_clip=0)

                    gen.reset(train_features)

                    gen.train(train_features)

                    syntheticPoints= gen.generateData(n_syn_samples)

                    syntheticPoints=pd.DataFrame(syntheticPoints, columns=ordered_features)


                    # Balance the synthetic data
                    balanced_synthetic_data = balanced_syn_data(syntheticPoints, info['target_value_counts'], target)

                    # Create path
                    syn_save_path = os.path.join(synth_directory,"NextConvGeN", folder, task, "synthetic_data.csv")
                    balanced_synthetic_data_save_path = os.path.join(balanced_synth_directory,"NextConvGeN", folder, task, "synthetic_data.csv")

                    # Create the directories if they don't exist
                    os.makedirs(os.path.dirname(syn_save_path), exist_ok=True)
                    os.makedirs(os.path.dirname(balanced_synthetic_data_save_path), exist_ok=True)

                    syntheticPoints.to_csv(syn_save_path, index=False)
                    balanced_synthetic_data.to_csv(balanced_synthetic_data_save_path, index=False)

                    print("Generated Synthetic data for {} and stored in path sucessfully".format(folder))

                else:
                    print(f"'additional_info.json' file not found in '{task_directory}'")

            else:
                print(f"'training_data.csv' not found in '{task_directory}'")

        else:
            print(f"'semi-supervised' directory not found in '{folder}'")



def CTGAN_pipeline(task = "semi-supervised", base_directory, synth_directory, balanced_synth_directory):
    # List all subdirectories (folders) in the base directory
    all_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

    # Iterate through all the folders
    for folder in all_folders:
        folder_path = os.path.join(base_directory, folder)

        # Check if the 'semi-supervised' directory exists in the current folder
        task_directory = os.path.join(folder_path, task)
        if os.path.exists(task_directory):

            # Check if 'training_data.csv' exists in the 'semi-supervised' directory
            training_data_path = os.path.join(task_directory, "training_data.csv")
            if os.path.exists(training_data_path):

                # Load the training data
                data = pd.read_csv(training_data_path)

                # Load additional info from the 'additional_info.json' file in the 'semi-supervised' directory
                info_path = os.path.join(task_directory, "additional_info.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r') as info_file:
                        info = json.load(info_file)

                    # Extract information from the info dictionary
                    categorical_columns = []
                    integer_columns = []

                    if info['indices_ordinal_features'] is not None:
                        categorical_columns.extend([info['ordered_features'][i] for i in info['indices_ordinal_features']])

                    if info['indices_nominal_features'] is not None:
                        categorical_columns.extend([info['ordered_features'][i] for i in info['indices_nominal_features']])

                    if info['indices_continuous_features'] is not None:
                        integer_columns.extend([info['ordered_features'][i] for i in info['indices_continuous_features']])
                    
                    if isinstance(info.get("target"), str):
                        target=info['target']
                    else:
                        target=info['target'][0]
                    
                    # Add the target column
                    if info['target'] is not None:
                        categorical_columns.append(target)
                    
                    MetaData = meta_data(data, categorical_columns, integer_columns)
                    Meta_Data= SingleTableMetadata.load_from_dict(MetaData)


                    # Create the synthesizer with the extracted information
                    ctgan=CTGANSynthesizer(Meta_Data)

                    # Fit the synthesizer
                    ctgan.fit(data)

                    # Generate synthetic samples
                    syn = ctgan.sample(num_rows=data.shape[0]*5)
                    
                    syn[categorical_columns]=syn[categorical_columns].astype(float).astype(int)

                    # Balance the synthetic data
                    balanced_synthetic_data = balanced_syn_data(syn, info['target_value_counts'], target)

                    # Create path
                    syn_save_path = os.path.join(synth_directory, "CTGAN", folder, task, "synthetic_data.csv")
                    balanced_synthetic_data_save_path = os.path.join(balanced_synth_directory, "CTGAN", folder, task, "synthetic_data.csv")

                    # Create the directories if they don't exist
                    os.makedirs(os.path.dirname(syn_save_path), exist_ok=True)
                    os.makedirs(os.path.dirname(balanced_synthetic_data_save_path), exist_ok=True)

                    syn.to_csv(syn_save_path, index=False)
                    balanced_synthetic_data.to_csv(balanced_synthetic_data_save_path, index=False)
                    
                    print("Generated Synthetic data for {} and stored in path sucessfully".format(folder))  

                else:
                    print(f"'additional_info.json' file not found in '{task_directory}'")

            else:
                print(f"'training_data.csv' not found in '{task_directory}'")

        else:
            print(f"'semi-supervised' directory not found in '{folder}'")
            

            
            
            
def reformat_syn_data(task = "semi-supervised", base_directory, synth_directory, balanced_synth_directory):
    syn_array_directory = 'TabDDPM_privacy_syn_data'
    # List all subdirectories (folders) in the base directory
    all_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

    # Iterate through all the folders
    for folder in all_folders:
        folder_path = os.path.join(base_directory, folder)
        syn_array_path = os.path.join(syn_array_directory, folder, task)

        # Check if the 'task' directory exists in the current folder
        task_directory = os.path.join(folder_path, task)
        print(task_directory)
        if os.path.exists(task_directory) and os.path.exists(syn_array_path):
            print('True')
            
            X_num_path = os.path.join(syn_array_path, "X_num_train.npy")
            X_cat_path = os.path.join(syn_array_path, "X_cat_train.npy")
            y_train_path = os.path.join(syn_array_path, "y_train.npy")

            # Load the data
            X_num = np.load(X_num_path, allow_pickle=True)
            X_cat = np.load(X_cat_path, allow_pickle=True)
            y_train = np.load(y_train_path, allow_pickle=True)
            y_train = y_train[:,np.newaxis]
            
            print(X_num.shape) 
            print(X_cat.shape) 
            print(y_train.shape) 
            
            syn_array = np.concatenate((X_num, X_cat, y_train), axis=1)
            

            # Load additional info from the 'additional_info.json' file in the 'semi-supervised' directory
            info_path = os.path.join(task_directory, "additional_info.json")
            if os.path.exists(info_path):
                with open(info_path, 'r') as info_file:
                    info = json.load(info_file)

                # Extract information from the info dictionary
                ordered_features = info['ordered_features']

                if isinstance(info.get("target"), str):
                    target=info['target']
                else:
                    target=info['target'][0]

                # Add the target column
                if info['target'] is not None:
                    ordered_features.append(target)
                
                syn_df = pd.DataFrame(syn_array, columns = ordered_features)


                # Balance the synthetic data
                balanced_synthetic_data = balanced_syn_data(syn_df, info['target_value_counts'], target)

                # Create path
                syn_save_path = os.path.join(synth_directory,"TabDDPM", folder, task, "synthetic_data.csv")
                balanced_synthetic_data_save_path = os.path.join(balanced_synth_directory,"TabDDPM", folder, task, "synthetic_data.csv")

                # Create the directories if they don't exist
                os.makedirs(os.path.dirname(syn_save_path), exist_ok=True)
                os.makedirs(os.path.dirname(balanced_synthetic_data_save_path), exist_ok=True)

                syn_df.to_csv(syn_save_path, index=False)
                balanced_synthetic_data.to_csv(balanced_synthetic_data_save_path, index=False)

                print("Formatted Synthetic data for {} and stored in path sucessfully".format(folder))

            else:
                print(f"Directory not found in '{task_directory}'")

        else:
            print(f"Directory not found in '{folder}'")



            
            
""""            
def TVAE_pipeline(task = "semi-supervised", base_directory, synth_directory, balanced_synth_directory):
    # List all subdirectories (folders) in the base directory
    all_folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

    # Iterate through all the folders
    for folder in all_folders:
        folder_path = os.path.join(base_directory, folder)

        # Check if the 'semi-supervised' directory exists in the current folder
        task_directory = os.path.join(folder_path, task)
        if os.path.exists(task_directory):

            # Check if 'training_data.csv' exists in the 'semi-supervised' directory
            training_data_path = os.path.join(task_directory, "training_data.csv")
            if os.path.exists(training_data_path):

                # Load the training data
                data = pd.read_csv(training_data_path)

                # Load additional info from the 'additional_info.json' file in the 'semi-supervised' directory
                info_path = os.path.join(task_directory, "additional_info.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r') as info_file:
                        info = json.load(info_file)

                    # Extract information from the info dictionary
                    categorical_columns = []
                    integer_columns = []

                    if info['indices_ordinal_features'] is not None:
                        categorical_columns.extend([info['ordered_features'][i] for i in info['indices_ordinal_features']])

                    if info['indices_nominal_features'] is not None:
                        categorical_columns.extend([info['ordered_features'][i] for i in info['indices_nominal_features']])

                    if info['indices_continuous_features'] is not None:
                        integer_columns.extend([info['ordered_features'][i] for i in info['indices_continuous_features']])
                    
                    if isinstance(info.get("target"), str):
                        target=info['target']
                    else:
                        target=info['target'][0]
                    
                    # Add the target column
                    if info['target'] is not None:
                        categorical_columns.append(target)
                    
                    MetaData = meta_data(data, categorical_columns, integer_columns)
                    Meta_Data= SingleTableMetadata.load_from_dict(MetaData)


                    # Create the synthesizer with the extracted information
                    tvae=TVAESynthesizer(Meta_Data)

                    # Fit the synthesizer
                    tvae.fit(data)

                    # Generate synthetic samples
                    n_syn_samples=data.shape[0]*5
                    syn = tvae.sample(num_rows=n_syn_samples)
                    
                    syn[categorical_columns]=syn[categorical_columns].astype(float).astype(int)

                    # Balance the synthetic data
                    #balanced_synthetic_data = balanced_syn_data(syntheticPoints, info['target_value_counts'], target)
                    max_attempts = 5  # Adjust this as needed
                    attempts = 0

                    while attempts < max_attempts:
                        try:
                            # Attempt to balance the synthetic data
                            balanced_synthetic_data = balanced_syn_data(syn, info['target_value_counts'], target)
                            break  # Break out of the loop if successful
                        except ValueError as e:
                            # Handle the ValueError (sampling size larger than population) here
                            print(f"Error: {e}")

                            # Adjust the number of synthetic samples and regenerate data
                            n_syn_samples *= 2  
                            syn = tvae.sample(num_rows=n_syn_samples)

                            #syn = pd.DataFrame(syntheticPoints, columns=ordered_features)

                            attempts += 1

                    if attempts == max_attempts:
                        print("Maximum attempts reached. Error not resolved.")
                    else:
                        print("Balancing successful after", attempts, "attempts.")

                    # Create path
                    syn_save_path = os.path.join("SyntheticData", "TVAE", folder, task, "synthetic_data.csv")
                    balanced_synthetic_data_save_path = os.path.join("BalancedSyntheticData", "TVAE", folder, task, "synthetic_data.csv")

                    # Create the directories if they don't exist
                    os.makedirs(os.path.dirname(syn_save_path), exist_ok=True)
                    os.makedirs(os.path.dirname(balanced_synthetic_data_save_path), exist_ok=True)

                    syn.to_csv(syn_save_path, index=False)
                    balanced_synthetic_data.to_csv(balanced_synthetic_data_save_path, index=False)
                    
                    print("Generated Synthetic data for {} and stored in path sucessfully".format(folder))  

                else:
                    print(f"'additional_info.json' file not found in '{task_directory}'")

            else:
                print(f"'training_data.csv' not found in '{task_directory}'")

        else:
            print(f"'semi-supervised' directory not found in '{folder}'")
            
            
""""          
            
