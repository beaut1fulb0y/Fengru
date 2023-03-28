import os
import shutil


def rename(base_path):
    for root, dir, files in os.walk(base_path):
        old_root = root
        new_root = old_root.replace('未标注', 'unlabeled')
        new_root = new_root.replace('未患病', 'unafflicted')
        new_root = new_root.replace('标注', 'labeled')
        new_root = new_root.replace('患病', 'afflicted')

        os.rename(root, new_root)


def rename1():
    base_path = '../data'
    data_folders = ['data1', 'data2']
    label_folders = ['labeled', 'unlabeled']
    condition_folders = ['afflicted', 'unafflicted']
    target_subfolder_numbers = [1, 2, 3]

    for data_folder in data_folders:
        for label_folder in label_folders:
            for condition_folder in condition_folders:
                folder_path = os.path.join(base_path, data_folder, label_folder, condition_folder)
                patient_folders = get_subfolder_paths(folder_path)
                for patient_folder in patient_folders:
                    patient = patient_folder.split('\\')[-1]
                    old_folder_structure = os.path.join(base_path, data_folder, label_folder, condition_folder, patient)
                    if os.path.exists(old_folder_structure):
                        # print(new_folder_structure)
                        for entry in os.scandir(old_folder_structure):
                            for file_entry in os.scandir(entry.path):
                                file_name = file_entry.path.split('\\')[-1]
                                target_subfolder_number = entry.path.split('\\')[-1]
                                new_folder_structure = os.path.join(base_path, data_folder,
                                                                    target_subfolder_number, label_folder,
                                                                    condition_folder, patient)
                                os.makedirs(new_folder_structure, exist_ok=True)
                                shutil.move(file_entry.path, new_folder_structure)
                    # print(os.path.join(patient_folder, str(target_subfolder_number)))


def remove_files(base_path):
    for entry in os.scandir(base_path):
        if entry.is_file():
            # print(entry.path)
            os.remove(entry.path)


def get_subfolder_paths(folder_path):
    subfolder_paths = []
    for entry in os.scandir(folder_path):
        if entry.is_dir():
            subfolder_paths.append(entry.path)
    return subfolder_paths


def remove():
    base_path = '../data'
    data_folders = ['data1', 'data2']
    label_folders = ['labeled', 'unlabeled']
    condition_folders = ['afflicted', 'unafflicted']
    target_subfolder_numbers = [1, 2, 3]

    for data_folder in data_folders:
        for label_folder in label_folders:
            for condition_folder in condition_folders:
                folder_path = os.path.join(base_path, data_folder, label_folder, condition_folder)
                patient_folders = get_subfolder_paths(folder_path)
                for patient_folder in patient_folders:
                    if os.path.exists(patient_folder):
                        # print(patient_folder)
                        remove_files(patient_folder)
                        pass


if __name__ == '__main__':
    rename('../data')
    # remove()
    # rename1()
    pass
