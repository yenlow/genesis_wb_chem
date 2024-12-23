# Databricks notebook source
# MAGIC %md
# MAGIC MolXPT OOTB is easy to use but performance cannot be reproduced as stated in [paper arxiv: 2305.10688v2](https://arxiv.org/pdf/2305.10688v2) and on [leaderboard](https://paperswithcode.com/paper/molxpt-wrapping-molecules-with-text-for)
# MAGIC
# MAGIC Suspect it needs additional pretraining

# COMMAND ----------

# MAGIC %pip install -U sacremoses rdkit pubchempy matplotlib
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip freeze > requirements_load_molxpt.txt

# COMMAND ----------

from transformers import AutoTokenizer, BioGptForCausalLM
from rdkit import Chem
from rdkit.Chem import Draw
import torch
import mlflow
import pubchempy as pcp
import matplotlib.pyplot as plt

# COMMAND ----------

# TODO: put into config dict when available
catalog = "genesis-workbench"
schema = "dev_chem"
token_start = '<start-of-mol>'
token_end = '<end-of-mol>'

# https://huggingface.co/zequnl/molxpt
model_extsrc = "zequnl/molxpt"
# if register model on UC
#registered_model_name = f"{catalog}.{schema}.molxpt"
# if register on legacy WS
registered_model_name = "molxpt"

# COMMAND ----------

# TODO: register model to Workspace until graph signatures are allowed 
#mlflow.set_registry_uri("databricks-uc")
#registered_model_name = f"{catalog}.{schema}.molxpt"
mlflow.set_registry_uri("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download models

# COMMAND ----------

model = BioGptForCausalLM.from_pretrained(model_extsrc)
molxpt_tokenizer = AutoTokenizer.from_pretrained(model_extsrc, trust_remote_code=True)

# COMMAND ----------

# Helper function to decode tensor to text
def decode_output(output):
    return [molxpt_tokenizer.decode(i) for i in output]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Generate molecules (Text to molecules)

# COMMAND ----------

# Aspirin as starting molecule
input_smile = "CC(=O)OC1=CC=CC=C1C(=O)O"
#input_smile = "c1cnc2c(c1)ccc1cccnc12"

# Get compound name from SMILES
cpds = pcp.get_compounds(input_smile, 'smiles')
fig = Draw.MolToMPL(Chem.MolFromSmiles(input_smile))
plt.title(cpds[0].synonyms[0])
plt.axis('off')
plt.show()

# COMMAND ----------

input_example = molxpt_tokenizer(f'<start-of-mol>{input_smile}<end-of-mol> is ', return_tensors="pt").input_ids
num_molecules = 3
input_example

# COMMAND ----------

# MAGIC %md
# MAGIC #### Using model in memory

# COMMAND ----------

output = model.generate(
    input_example,
    max_new_tokens=300,
    num_return_sequences=num_molecules,
    temperature=0.75,
    top_p=0.95,
    do_sample=True,
)
results_text = decode_output(output)
results_text

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log model to mlflow

# COMMAND ----------

#pipeline = pipeline(model=model, tokenizer=molxpt_tokenizer)
components = {
    "model": model,
    "tokenizer": molxpt_tokenizer,
}

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.transformers.log_model(
        transformers_model=components,
        artifact_path="model",
        register_model=registered_model_name,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load model and tokenizer from mlflow

# COMMAND ----------

run_id = run.info.run_id
#run_id = "b003c0da911c4737a5097f1a5f0acc21"
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.transformers.load_model(model_uri)
run_id

# COMMAND ----------

# Explicitly register model as mlflow.transformers.log_model may not always automatically register models
mlflow.register_model(model_uri, registered_model_name)

# COMMAND ----------

# Using model loaded from mlflow
output_reloaded = loaded_model.model.generate(
    input_example,
    max_new_tokens=300,
    num_return_sequences=num_molecules,
    temperature=0.75,
    top_p=0.95,
    do_sample=True,
)
results_text = decode_output(output_reloaded)
results_text

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Text to molecule

# COMMAND ----------

txt2mol = molxpt_tokenizer(f'What is a non-penicillin based antibiotic? Complete the following sentence with a SMILE and nothing else. The compound is {token_start}', return_tensors="pt").input_ids
num_molecules = 3
output = model.generate(
    txt2mol,
    max_new_tokens=300,
    num_return_sequences=num_molecules,
    temperature=0.75,
    top_p=0.95,
    do_sample=True,
)
results_text = decode_output(output)
results_text

# COMMAND ----------

# Prompt is taken verbatim from paper but response does not match paper
txt2mol = molxpt_tokenizer(f'The molecule is a bile acid taurine conjugate of ursocholic acid. It has a role as a human metabolite and a rat metabolite. It derives from ursocholic acid. It is a conjugate acid of taurousocholate.', return_tensors="pt").input_ids
num_molecules = 3
output = model.generate(
    txt2mol,
    max_new_tokens=300,
    num_return_sequences=num_molecules,
    temperature=0.1,
    top_p=0.95,
    do_sample=True,
)
results_text = decode_output(output)
results_text

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Predict classes
# MAGIC The following predicts binary endpoints (e.g. BBB) from SMILES. Poor performance suggests they need additional FT

# COMMAND ----------

mol2bbb = molxpt_tokenizer(f'We can conclude that the blood brain barrier penetration of {token_start}{input_smile}{token_start} true/false. Answer only true or false and nothing else.', return_tensors="pt").input_ids
num_molecules = 10
output = model.generate(
    mol2bbb,
    max_new_tokens=100,
    num_return_sequences=num_molecules,
    temperature=0.75,
    top_p=0.95,
    do_sample=True,
)
results_sider = decode_output(output)
results_sider

# COMMAND ----------

mol2sider = molxpt_tokenizer(f'We can conclude that the {token_start}{input_smile}{token_start} can bring about the side effect of Peptic ulcer is true/false', return_tensors="pt").input_ids
num_molecules = 10
output = model.generate(
    mol2sider,
    max_new_tokens=100,
    num_return_sequences=num_molecules,
    temperature=0.75,
    top_p=0.95,
    do_sample=True,
)
results_sider = decode_output(output)
results_sider

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: SMILES to embeddings

# COMMAND ----------

smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Oc1ccccc1C(=O)O"]
inputs = molxpt_tokenizer(smiles, padding=True, return_tensors="pt")
inputs

# COMMAND ----------

with torch.no_grad():
    outputs = model(**inputs)
outputs

# COMMAND ----------


