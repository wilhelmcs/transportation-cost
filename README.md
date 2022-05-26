# transportation-cost
Transportation cost problem solver using Russell, Vogel and North West Corner approximation methods

## Installation

Download the GitHub repository via SSH or HTTPS

```bash
$ git clone URL
````

Define a virtual environment

```bash
$ python3 -m venv .venv
```

Activate the virtual environment for further dependencies.

```bash
$ source .venv/bin/activate
```

Make sure pip is installed on your computer, check the [pip official documentation](https://pip.pypa.io/en/stable/installation/)
if it isn't.

```bash
(.venv) $ pip3 install -r requirements.txt
```

## Run

```bash
(.venv) $ ./transporte.py [-h] method file.txt
```

### Arguments

`method` Approximation method used to solve the problem.

    1 = NORTH WEST APPROXIMATION METHOD
    2 = VOGEL APPROXIMATION METHOD
    3 = RUSSELL APPROXIMATION METHOD    

`file.txt` Text file with the transportation problem in the correct format.
 The file has the following structure, separated by commas.
Supply column, demand row, transportation costs. 

For example if the problem comes in the following form:
           
            D1      D2      D3      Supply
    S1      8       6       10      2000
    S2      10      4       9       2500
    Demand 1500     2000    1000
        
The file must come as shown below:

    2000,2500
    1500,2000,1000
    8,6,10
    10,4,9

The `-h` flag displays the help for the program execution. test

---
## Authors

* Wilhelm Carstens **@wolam**
