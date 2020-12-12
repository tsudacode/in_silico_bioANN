# in_silico_bioANN

*In silico* recurrent neural network model of biological neural networks used in:

Trujillo CA, Adams JW, Negraes PD, Carromeu C, Tejwani L, Acab A, Tsuda B, Thomas CA, Sodhi N, Fichter KM, Romero S, Zanella F, Sejnowski TJ, Ulrich H, Muotri AR. Pharmacological reversal of synaptic and network pathology in human MECP2-KO neurons and cortical organoids. *EMBO Molecular Medicine*, e12523, 2020.

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

# Citation

If you use this repo in your research, please cite:

    @article{Trujillo_2020,
    Author = {Trujillo CA and Adams JW and Negraes PD and Carromeu C and Tejwani L and Acab A and Tsuda B and Thomas CA and Sodhi N and Fichter KM and Romero S and Zanella F and Sejnowski TJ and Ulrich H and Muotri AR},  
    Title = {Pharmacological reversal of synaptic and network pathology in human MECP2-KO neurons and cortical organoids},  
    Journal = {EMBO Molecular Medicine},  
    Pages = {e12523},  
    DOI = {https://doi.org/10.15252/emmm.202012523},  
    Year = {2020}}
