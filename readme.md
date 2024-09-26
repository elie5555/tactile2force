## Virtual Environment and Dependencies Installation

# Python --version == Python 3.10.12

Navigate to the project directory.

```
cd interaction-forces-estimation-from-magnetic-tactile-sensors/
```

Create a virtual environment using venv.

```
python3 -m venv env
```

This will create a new directory called env containing the virtual environment.

4. Activate the virtual environment.

* On Windows:

```
env\Scripts\activate 
```

* On macOS/Linux:

```
source env/bin/activate 
```

5. Install the required packages using pip.

```
pip install -r requirements.txt
```

This will install all the packages listed in the requirements.txt file.
6. Verify that the installation was successful by running a script or importing a module in the virtual environment.