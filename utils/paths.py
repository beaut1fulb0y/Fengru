import os

import pandas as pd


def create_data_paths(root: str):
    data_folder_name = []
    data_folder_id = []
    label_status = []
    view = []
    afflicted_status = []
    patient = []
    image = []
    for root_dir, _, files in os.walk(root, topdown=True):
        for file in files:
            if file[-4: -1] + file[-1] == '.jpg':
                string = os.path.join(root_dir, file)
                string_list = string.split('/')
                data_folder_name.append(string_list[1])
                data_folder_id.append(string_list[2])
                label_status.append(string_list[3])
                view.append(string_list[4])
                afflicted_status.append(string_list[5])
                patient.append(string_list[6])
                image.append(string_list[7])

    my_data = {'root': data_folder_name, 'folder_id': data_folder_id, 'label': label_status, 'view': view,
               'afflict': afflicted_status, 'patient': patient, 'image': image}
    df = pd.DataFrame(my_data)
    df.to_csv('../dataset/mydata.csv', index=False)
    print(df.head(5))



if __name__ == '__main__':
    create_data_paths('../data')
