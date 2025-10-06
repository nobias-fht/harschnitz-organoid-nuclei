import os
import pandas as pd
import easygui

file_path = easygui.diropenbox(title="Select folder where images are stored")

folders = os.listdir(file_path)

for i, folder in enumerate(folders):
    csv_path = os.path.join(file_path, folder, 'quantification', 'summary.csv')
    df = pd.read_csv(csv_path)

    if i == 0:
        label_list = df['label'].tolist()
        new_df = pd.DataFrame(columns=label_list)
        data_list = df['data'].tolist()
        new_df.loc[i] = data_list
    else:
        data_list = df['data'].tolist()
        new_df.loc[i] = data_list

new_df.to_csv(os.path.join(file_path, 'summary_all.csv'), index=False)



