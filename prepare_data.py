# Python script to unzip the data 
from zipfile import ZipFile

# uncompress the zip file 
def uncompress_file(zip):
    with ZipFile(zip) as zipf:
        zipf.extractall('/opt/udacity_training_data')

# Uncompress the         
uncompress_file('data.zip')