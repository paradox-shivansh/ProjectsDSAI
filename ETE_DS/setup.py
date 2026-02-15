from setuptools import setup, find_packages
from typing import List



HYPEN_E_DOT = "-e ."

# def get_reqirments(file_path):
#     with open(file_path) as f:
#         req = f.read().splitlines()
        
#     if HYPEN_E_DOT in req:
#         req.remove(HYPEN_E_DOT)
        
        
#     return req


# def get_reqirments(file_path:str)->list[str]:
#     with open(file_path) as f:
#        requirments=f.readlines()
#        requirments=[req.replace("\n","") for req in requirments] 


# BEST METHOD TO GET REQUIRMENTS FROM A FILE
def get_reqirments(file_path: str) -> List[str]:
    with open(file_path) as f:
        req = f.read().splitlines()
        
    if HYPEN_E_DOT in req:
        req.remove(HYPEN_E_DOT)
        
        
    return req



setup(
    name='ete_ds',
    version='0.0.1',
    author='Shivansh Srivsastava',
    author_email='theaverageguy19@gmail.com',
    description='A package for data science utilities',
    packages=find_packages(),
    # install_requires=['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn'],
    install_requires=get_reqirments('requirments.txt') ,
    
)