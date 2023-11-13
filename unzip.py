import requests
from datetime import timedelta,date,datetime
import time
import os
import zipfile


zip_file = zipfile.ZipFile('detection_dataset.zip')
zip_list = zip_file.namelist() # 压缩文件清单，可以直接看到压缩包内的各个文件的明细
for f in zip_list: # 遍历这些文件，逐个解压出来，
    zip_file.extract(f,'.')
zip_file.close() # 不能少！

