import boto3
import os

# https://stackoverflow.com/a/62945526
s3 = boto3.resource('s3') # assumes credentials & configuration are handled outside python in .aws directory or environment variables
def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory

    Parameters
    ----------
    bucket_name : str
        the name of the s3 bucket

    s3_folder: str
        the folder path in the s3 bucket

    local_dir: str
        a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)


def cp_from_s3(uri: str, default_loc="/data") -> str:
    """
    Checks the uri for the presence of s3:// and sets up a
    local folder from the s3 bucket otherwise returns the uri as is

    Parameters
    ----------
    uri : str
        The s3 bucket to clone locally

    Returns
    -------
    str
        location of the local folder
    """
    print(f"The URI is {uri}")
    if uri.startswith('s3://'):
        print(f"    which was identified as an s3 source")
        bucket_name = uri[5:].split('/')[0]
        s3_folder = '/'.join(uri[5:].split('/')[1:])
        destination_dir = f"{default_loc}/{uri[5:]}"
        download_s3_folder(bucket_name=bucket_name, s3_folder=s3_folder, local_dir=destination_dir)
        return destination_dir
    else:
        return uri


def cp_to_s3(local_dir, s3_dir):
    """
    Copies files from a local directory to an s3 bucket

    Parameters
    ----------
    local_dir : str
        The local directory to copy from
    s3_dir : str
        The s3 bucket to copy to
    """
    s3 = boto3.client('s3')
    bucket_name = s3_dir[5:].split('/')[0]
    prefix = s3_dir[5+len(bucket_name)+1:]
    for f in os.listdir(local_dir):
        key = f'{prefix}{"" if prefix.endswith("/") else "/"}{f}'
        s3.upload_file(f"{local_dir}/{f}", bucket_name, key)
