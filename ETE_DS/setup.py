from setuptools import setup, find_packages

def get_reqirments(file_path):
    with open(file_path) as f:
        req = f.read().splitlines()
    return req
    


setup(
    name='ete_ds',
    version='0.0.1',
    author='Shivansh Srivsastava',
    author_email='theaverageguy19@gmail.com',
    description='A package for data science utilities',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn'],
    install_requires=get_reqirments('requirements.txt') ,
    
)