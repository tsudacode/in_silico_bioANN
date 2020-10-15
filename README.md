# in_silico_bioANN

In silico recurrent neural network model of arbitrary biological neural network used in:  
Trujillo CA, Adams JW, Negraes PD, Carromeu C, Tejwani L, Acab A, Tsuda B, Thomas CA, Sodhi N, Fichter KM, Zanella F, Sejnowski TJ, Ulrich H, Muotri AR. *Pharmacological reversal of multiple phenotypes in human MECP2-KO neurons and networks.* In review. 2020.

Organization of **bioANN.py** is
  - helper fxns
  - definition of network class
  - definition of worker class
      - test fxn
  - main
      - definition of parameters and output directories
      - creation of central network
      - creation of training workers
      - creation of testing workers
      - script to deploy workers for training AND testing

Command to run bioANN:  
`python3 bioANN.py [NETSZ] [GPU] [RUNNO]`

# Citation

If you use this repo in your research, please cite:

    @article{Trujillo_2020,
    Author = {Trujillo, CA and Adams, Jason W and Negraes, PD and Carromeu, C and Tejwani, L and Acab, A and Tsuda, Ben and Thomas, CA and Sodhi, N and Fichter, KM and Zanella, F and Sejnowski, Terrence J and Ulrich, H and Muotri AR},  
    Journal = {In review},  
    Title = {Pharmacological reversal of multiple phenotypes in human MECP2-KO neurons and networks},  
    Year = {2020}}
