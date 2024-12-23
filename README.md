# genesis_wb_chem
Temporary repo for the small molecule models, data and workflows until access is granted to the main repo.

In this repo are 2 key chemical packages: `chemprop` and `molxpt`

## `chemprop`
A chemical property prediction package popular among pharma cheminformaticians.<br>
Ref: [Chemprop: a Machine Learning Package for Chemical Property Prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01250)<br>
This is the v2 followup to the original [Cell paper](https://www.cell.com/cell/fulltext/S0092-8674(20)30396-2) discovering a novel and structurally diverse antibiotic halicin


### Example NBs/workflows:
1. [Load data](chemprop/Load data): load example datasets (CSV) into Delta tables on Unity Catalog
#### Single-task training and inferencing
2. [Chemprop: fit ClinTox](chemprop/Chemprop:%20fit%20ClinTox.ipynb) classifier (single-task)
3. [Chemprop: inference ClinTox](chemprop/Chemprop:%20inference%20clintox.ipynb): use ClinTox classifier from NB #2 to predict ClinTox properties of DrugBank
#### Multi-task training and inferencing
4. [Chemprop: multitask training](chemprop/Chemprop:%20multitask%20training.ipynb): do multi-task regression on 10 continuous endpoints from [ADMET-AI](https://academic.oup.com/bioinformatics/article/40/7/btae416/7698030#469619671)
5. [Chemprop: multitask inference](chemprop/Chemprop:%20inference%20clintox.ipynb): use multi-task regressor from NB #4 to predict 10 continuous endpoints on DrugBank
#### Inferencing using Chemprop example models (.ckpt) and CSV
6. [Chemprop: inference reaction](chemprop/Chemprop:%20inference%20reaction.ipynb): use reaction regressor from [chemprop](https://github.com/chemprop/chemprop/tree/f8774bd92174f97030e5ba25eb971e33f45cb96b) to predict reactivity (Ea)
7. [Chemprop: multicomponent reaction](chemprop/Chemprop:%20inference%20reaction.ipynb): use multicomponent regressor from [chemprop](https://github.com/chemprop/chemprop/tree/f8774bd92174f97030e5ba25eb971e33f45cb96b) to predict solubility of compound (component 1) in solvent (component 2)

---------------------------------------
## `molxpt`
A multimodal model pretrained on text (PubMed) and molecular SMILES (PubChem) so you can prompt chemical generation tasks with natural language.<br>
Ref: [molXPT: Wrapping Molecules with Text for Generative Pre-training](https://arxiv.org/abs/2305.10688)

### Example NB:
Load molXPT for inferencing including text2molecule, molecule2text and molecular embeddings generation