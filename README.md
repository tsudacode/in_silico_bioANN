# in_silico_bioANN

*In silico* recurrent neural network model of biological neural networks.

Organization of **bioANN.py** is
  - definition of network class
  - definition of simulation class
      - test fxn
  - main
      - definition of parameters and output directory
      - creation of simulation of network
      - script to run simulation and gather data

Command to run bioANN:

`python3 bioANN.py [NETSZ] [P_CON] [P_INH] [W_MASK] [KD]`

where `[NETSZ]` is number of neurons in network, `[P_CON]` is connectivity, `[P_INH]` is fraction of network that are inhibitory neurons, `[W_MASK]` is the catergory of synapses to perturb, and `[KD]` is the synaptic perturbation factor.

