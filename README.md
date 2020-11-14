# Install Requirements (python 3)

pip3 install -r requirements.txt

# Initialize Model


```python
from node_model import *
import numpy as np
```


```python
model = Node_model(prior=(0,1),noise=(0,1))
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_3a0460d6844c38c2a414ece69540d921 NOW.
    

Node values will have a prior distribution of N(prior[0],prior[1]). Noise values will have an assumption distribution of N(noise[0],noise[1]).

Usually it is good enough to leave both settings to (0,1). 

# Read Graph

Graphs are required to have the following format as in *example_graph.txt*:

start end observed_value

Then simply use *model.read_graph(graph_dir)* to input the graph




```python
model.read_graph('example_graph.txt')
```

# Optimization


```python
model.optimize()
```




    OrderedDict([('v',
                  array([-0.85781401, -1.13165061, -0.64344908, -0.2928115 ,  0.30745006,
                         -0.71146706, -0.3863362 ,  1.28912426,  0.32361522,  0.84305583,
                          0.07027949, -0.56523112,  0.52385088,  0.22470481, -0.49363143,
                          0.84228893, -1.34628577,  1.34305531, -2.06039571,  1.45612158,
                         -0.66984386,  0.0580456 ,  1.03631945, -0.58235709,  0.32351126,
                          1.09900337]))])



# Prediction

Output prediction is in a dictionary format where each key is a tulpe representing the start and end point of each edge.


```python
model.predict_val()
```




    {('0', '1'): -0.27383660158660184,
     ('0', '2'): 0.214364933027403,
     ('0', '3'): 0.565002510399619,
     ('0', '4'): 1.165264068778154,
     ('0', '5'): 0.14634695378203544,
     ('0', '6'): 0.47147781524229876,
     ('0', '7'): 2.1469382721729264,
     ('0', '8'): 1.18142922816159,
     ('0', '9'): 1.700869839662421,
     ('0', '10'): 0.928093500078194,
     ('0', '11'): 0.2925828903681167,
     ('0', '12'): 1.3816648919818508,
     ('0', '13'): 1.0825188233015477,
     ('3', '10'): 0.36309098967857506,
     ('3', '14'): -0.20081992425470768,
     ('15', '16'): -2.1885747011285677,
     ('15', '17'): 0.5007663758191944,
     ('17', '12'): -0.819204428671403,
     ('17', '18'): -3.4034510232913098,
     ('17', '19'): 0.11306627626987864,
     ('17', '14'): -1.8366867345083424,
     ('19', '7'): -0.1669973247502059,
     ('19', '20'): -2.1259654459407464,
     ('19', '21'): -1.3980759885907024,
     ('4', '10'): -0.23717056869995973,
     ('4', '5'): -1.0189171149961185,
     ('4', '22'): 0.7288693926648401,
     ('2', '14'): 0.14981765311750833,
     ('2', '23'): 0.061091985400373194,
     ('11', '21'): 0.6232767179643134,
     ('11', '18'): -1.4951645930061728,
     ('11', '12'): 1.089082001613734,
     ('21', '7'): 1.2310786638404965,
     ('21', '20'): -0.7278894573500443,
     ('18', '1'): 0.9287451010514542,
     ('18', '12'): 2.584246594619907,
     ('18', '23'): 1.4780386210658323,
     ('5', '13'): 0.9361718695195123,
     ('5', '23'): 0.12910996464574076,
     ('5', '14'): 0.2178356323628759,
     ('5', '22'): 1.7477865076609584,
     ('5', '24'): 1.0349783201319038,
     ('25', '16'): -2.445289143204861,
     ('25', '24'): -0.7754921129964137,
     ('16', '6'): 0.9599495715368072,
     ('1', '6'): 0.7453144168289005,
     ('1', '22'): 2.1679700630295957,
     ('14', '9'): 1.3366872535175096,
     ('13', '8'): 0.09891040486004221,
     ('13', '6'): -0.611041008059249,
     ('13', '10'): -0.15442532322335367,
     ('8', '23'): -0.9059723097338137,
     ('9', '6'): -1.2293920244201222,
     ('12', '24'): -0.20033961806791156,
     ('20', '22'): 1.706163310460608,
     ('22', '24'): -0.7128081875290547}


