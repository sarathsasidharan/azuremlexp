#%%
import azureml.core

print("SDK version:", azureml.core.VERSION)

# %%
from azureml.core import Workspace
ws = Workspace.get(name="<WORKSPACE-NAME>", subscription_id='<SUBSCRIPTION_ID>', resource_group='<RESOURCE_GROUP>')
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Resource group: ' + ws.resource_group, sep = '\n')
# %%
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# choose a name for your cluster
cluster_name = "cpu-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', 
                                                                max_nodes=3)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)

# Use the 'status' property to get a detailed status for the current cluster. 
print(compute_target.status.serialize())

# %%
from azureml.core import Experiment
experiment_name = 'aml_onnx'

exp = Experiment(workspace=ws, name=experiment_name)

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
                    compute_target=compute_target,
                    entry_script='mnist.py',
                    use_gpu=False)

estimator.conda_dependencies.remove_conda_package('pytorch=0.4.0')
estimator.conda_dependencies.add_conda_package('pytorch-nightly')
estimator.conda_dependencies.add_channel('pytorch')                   

# %%
run = exp.submit(estimator)
run.wait_for_completion(show_output = True)

# %%
run.get_file_names()
model_path = os.path.join('outputs', 'mnist.onnx')
run.download_file(model_path, output_file_path=model_path)

# %%
model = run.register_model(model_name='mnist', model_path=model_path)
print(model.name, model.id, model.version, sep = '\t')



# %%
models = ws.models
for name, m in models.items():
    print("Name:", name,"\tVersion:", m.version, "\tDescription:", m.description, m.tags)


