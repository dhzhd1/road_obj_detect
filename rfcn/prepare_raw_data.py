import zipfile
import tarfile
import os
import urllib


class PrepareRawData:
    def __init__(self):
        self.file_list = []
        self.dataset_path = '../dataset/'
        if not os.path.isdir(self.dataset_path):
            print("Dataset folder {} is not existing. Creating...".format(self.dataset_path))
            os.mkdir(self.dataset_path)

    def download(self, file_list=None):
        if file_list is not None:
            for file_name in file_list:
                data_file = urllib.URLopener()
                print("Starting download data file from {}".format(file_list[file_name]))
                data_file.retrieve(file_list[file_name], os.path.join(self.dataset_path, file_name))
                print("File has been downloaded and saved under {}".format(os.path.join(self.dataset_path, file_name)))
                self.file_list.append(os.path.join(self.dataset_path, file_name))
        else:
            print("No file url has been input")

    def extract(self):
        for file_name in self.file_list:
            ext_name = str(file_name).split('.')[-1].lower()
            if ext_name == "zip":
                zip_file = zipfile.ZipFile(file_name, 'r')
                zip_file.extractall(self.dataset_path)
                zip_file.close()
                print("File {} has been extracted at {}".format(file_name, self.dataset_path))
            elif ext_name == "tgz":
                tar_file = tarfile.open(file_name)
                tar_file.extractall(self.dataset_path)
                tar_file.close()
                print("File {} has been extracted at {}".format(file_name, self.dataset_path))
            else:
                print("There no proper method to extract this type of file [{}]".format(file_name))


if __name__ == '__main__':
    file_list = {'training.zip': 'https://s3-us-west-2.amazonaws.com/us-office/competition/training.zip',
                 'testing.zip': 'https://s3-us-west-2.amazonaws.com/us-office/competition/testing.zip'}
    prep_data = PrepareRawData()
    prep_data.download(file_list)
    prep_data.extract()
