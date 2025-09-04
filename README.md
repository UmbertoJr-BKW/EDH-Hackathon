# EDIH-Hackathon
Can we empower communities to finally break the energy trilemma—maximizing renewables, increase autarky, and guaranteeing network reliability—all at once?






```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh
```

Say what it is needed to install correctly the Miniconda, then say to restart.




```bash
conda env create -f environment.yml 
```

in a new terminal.

```bash
conda activate edh-hackathon-env

conda install -c conda-forge ipykernel

python -m ipykernel install --user --name edh-hackathon-env --display-name "EDH Hackathon (Conda)"

```


```bash



python -m ipykernel install --user --name edh-hackathon-env --display-name "EDH Hackathon (Conda)"


```
