#%%
import azureml.core

print("SDK version:", azureml.core.VERSION)

# %%
from azureml.core import Workspace
ws = Workspace.get(name="WS03051110", subscription_id='5c667bbb-a09e-4d96-bfe6-6659ade1e2cc', resource_group='WS03051110-rg2')
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Resource group: ' + ws.resource_group, sep = '\n')
# %%
from azureml.core.runconfig import RunConfiguration

# Edit a run configuration property on the fly.
run_local = RunConfiguration()

run_local.environment.python.user_managed_dependencies = True

# %%
from azureml.core import Experiment
experiment_name = 'aml_onnx'

exp = Experiment(workspace=ws, name=experiment_name)
# from azureml.core import ScriptRunConfig
# import os 

# script_folder = os.getcwd()
# src = ScriptRunConfig(source_directory = script_folder, script = 'train.py', run_config = run_local)
# run = exp.submit(src)
# run.wait_for_completion(show_output = True)

# %%

project_folder = './pytorch-mnist'
os.makedirs(project_folder, exist_ok=True)

# %%
import shutil
shutil.copy('mnist.py', project_folder)

# %%
from azureml.train.dnn import PyTorch

estimator = PyTorch(source_directory=project_folder, 
                    script_params={'--output-dir': './outputs'},
                    compute_target=run_local,
                    entry_script='mnist.py',
                    use_gpu=False)

estimator.conda_dependencies.remove_conda_package('pytorch=0.4.0')
estimator.conda_dependencies.add_conda_package('pytorch-nightly')
estimator.conda_dependencies.add_channel('pytorch')                   

# %%
run = exp.submit(estimator)
run.wait_for_completion(show_output = True)
#print(run.get_details())

# %%
