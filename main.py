from file_processing import read_vectors_db, create_db_from_files
import os

file_path ="data/original_data"
db_directory = "data/vectorDB"


"""
# create db
for file_name in os.listdir(file_path):
    create_db_from_files(file_name=file_name, file_path=file_path, description="file_name", db_directory=db_directory)

"""

for file_name in os.listdir(db_directory):
    #print(file_name)
    persist_path = os.path.join(db_directory, file_name)
    db = read_vectors_db(db_directory=persist_path)
    print(db._collection.count())