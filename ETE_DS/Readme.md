#END TO END DATASCIENCE PROJECT

1. Make and set a github repository (working on main branch).

2. Make a conda environment, here in have used python == 3.10.19 version.

3. Create a __init__.py for the system to concider the src folder as a 
package , it is generally empty and just requiered to make a ete project. (Internal folder behaves as a package.)

4. Make a fuction in (setup.py) for the installation and importing packages from the (Requirement.txt).

5. In requirments.txt (-e .) helps to trigger the setup file [editable mode] 
    (Python installs your project as a link, not a copy
    If you modify your source code, changes work immediately
    You don’t need to reinstall every time.)
    "pip install -r requirments.txt"

6. Creating a components folder in src to make the project more robust.

7. Creating a pipeline folder in src.

8. Creating all file in the src folder.

9. Making the exception.py file in src
    using the [sys] for the exception haldeling
    In end-to-end ML projects, sys is mainly used in the exception handling file to get detailed error information like:

    1. error message
    2. file name
    3. line number
    4. exact location of failure
    5. This helps in debugging large pipelines.

    [ remember Exception is a built in class in python]


10. Creating a logging.py file
    The logging module in Python is used to record events, errors, and messages while a program is running. It is the standard way to track program execution, especially in ML pipelines and production systems.

11. Completing EDA 

12. COMPLETING MODEL TRAINING
