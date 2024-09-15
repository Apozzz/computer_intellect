import os
import zipfile
import py7zr

def extract_zip(file_path, extract_to, file_to_extract=None):
    """
    Extracts a specific file from a .zip archive to the specified directory.
    If file_to_extract is provided, only that file is extracted.
    Deletes the zip file afterward.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        if file_to_extract:
            for file in zip_ref.namelist():
                if file_to_extract in file:
                    zip_ref.extract(file, extract_to)
                    print(f"Extracted {file} from {file_path} to {extract_to}")
        else:
            zip_ref.extractall(extract_to)
            print(f"Extracted all files from {file_path} to {extract_to}")
    
    os.remove(file_path)
    print(f"Deleted zip file: {file_path}")


def extract_7z(file_path, extract_to):
    """
    Extracts a .7z file to the specified directory.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with py7zr.SevenZipFile(file_path, mode='r') as archive:
        archive.extractall(path=extract_to)
    print(f"Extracted {file_path} to {extract_to}")

    os.remove(file_path)
    print(f"Deleted .7z file: {file_path}")
