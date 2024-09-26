# tactile2force

This repository contains to code that was used to benchmark models that map tactile signal to contact force for the Xela Robotics uSkin sensors mounted on the Allegro Hand robot. It also contains some datasets for the index, middle and ring fingertip arrays and a middlefinger square arrays.

## Docker

Install docker engine
Run the *launch_container.sh* script:


```sh launch_script.sh```

use ```python3``` to run individual model trainnings (*m0.py* to *m8.py*) or all models evaluation with *autotest.py*.

## General information

Results are recorded in the *results* folder. 

*pickle_data* contains the dataset. The dataset is composed of sequences of rotated force label in the common taxel frame *force_label.pkl*, tactile signal called *tact_data.pkl*, a common time vector called *time.pkl* and a metadata.json file that contains 0 offset for each force axis and for each taxel axis.

Fingertip arrays contains 30 3 axis taxels. Square arrays contains 16 3 axis taxels.



## 