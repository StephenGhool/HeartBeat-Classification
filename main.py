import pandas as pd

# read in the csv file that classifies the dataset
Set_A_Data = pd.read_csv("set_a.csv")
Set_B_Data = pd.read_csv("set_b.csv")

# display the first 5 and last 5 examples
# print(Set_A_Data.head(-1))

# checking to see how much null data we have
# print(Set_A_Data.isnull().sum())

# checking the number of different values in "label"
print(Set_A_Data["label"].value_counts())

# repeat for dataset B to have an idea of the data

# display the first 5 and last 5 examples
# print(Set_B_Data.info)

# checking to see how much null data we have
# print("Null in B:\n", Set_B_Data.isnull().sum())

# checking the number of different values in "label"
# print("Values in label for B:\n", Set_B_Data["label"].value_counts())

# this ensures that the audio file name is the same in both the csv file and wav file
Set_B_Data['fname']=Set_B_Data['fname'].str.replace('Btraining_', '')
Set_B_Data['fname']=Set_B_Data['fname'].str.replace('set_b/normal_', 'set_b/normal__')
Set_B_Data['fname']=Set_B_Data['fname'].str.replace('set_b/normal__noisynormal', 'set_b/normal_noisynormal')
Set_B_Data['fname']=Set_B_Data['fname'].str.replace('set_b/murmur_', 'set_b/murmur__')
Set_B_Data['fname']=Set_B_Data['fname'].str.replace('set_b/murmur__noisymurmur', 'set_b/murmur_noisymurmur')
Set_B_Data['fname']=Set_B_Data['fname'].str.replace('set_b/extrastole_', 'set_b/extrastole__')


# Concatenating the Dataset...which is joining both datasets vertically
Data = [Set_A_Data, Set_B_Data]
Data_concat = pd.concat(Data)
# print(Data_concat.head(-1))

# how much data is null in the Data_concat
# print("Total null values in Data_concat:\n", Data_concat.isnull().sum())

# we could see that sublabel and dataset columns does not contain any relevant information
cols_drop = ["sublabel", "dataset"]
Data_concat.drop(cols_drop, axis=1, inplace=True)
# print(Data_concat.head(-1))

# remove NaN rows
Data_concat.dropna(inplace=True)
# print(Data_concat.head(-1))

# reset the indexing of example such that they can be incremented
Data_concat.reset_index(inplace=True)
# print(Data_concat.head(-1))
print(Data_concat.isnull().sum())

# linking the audio files to the present dataset
Path_Wav_List = []
Category_List = []

# iterate over the total dataset and store the file name
for path_number in range(585):
    File_Path_Name = "audio/" + str(Data_concat["fname"][path_number])
    Path_Wav_List.append(File_Path_Name)
    Category_List.append(Data_concat["label"][path_number])

# let us create a series of the path and name it WAV
file_path_series = pd.Series(Path_Wav_List, name="WAV").astype(str)
# print(file_path_series)

# create a series of the category and name it CAT
category_series = pd.Series(Category_List, name="CAT")

# create the main dataset by combining the both
heartbeat_data = pd.concat([file_path_series, category_series], axis=1)

# print(heartbeat_data)

# let us shuffle the dataset. Setting frac to 1 so that 100% of the data is shuffled
heartbeat_data = heartbeat_data.sample(frac=1)
heartbeat_data.reset_index(inplace=True, drop=True)
# print(heartbeat_data["CAT"].value_counts())
print(heartbeat_data.head(-1))

# save the files
heartbeat_data.to_csv("Main_dataset.csv")
