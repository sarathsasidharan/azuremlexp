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


# %%
%%writefile score.py
import json
import time
import sys
import os
from azureml.core.model import Model
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime

def init():
    global session
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'mnist.onnx')
    session = onnxruntime.InferenceSession(model)

def preprocess(input_data_json):
    # convert the JSON data into the tensor input
    return np.array(json.loads(input_data_json)['data']).astype('float32')

def postprocess(result):
    # We use argmax to pick the highest confidence label
    return int(np.argmax(np.array(result).squeeze(), axis=0))

def run(input_data_json):
    try:
        start = time.time()   # start timer
        input_data = preprocess(input_data_json)
        input_name = session.get_inputs()[0].name  # get the id of the first input of the model   
        result = session.run([], {input_name: input_data})
        end = time.time()     # stop timer
        return {"result": postprocess(result),
                "time": end - start}
    except Exception as e:
        result = str(e)
        return {"error": result}

# %%
from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies.create(pip_packages=["numpy","onnxruntime","azureml-core", "azureml-defaults"])

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())   

# %%    
# 
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment


myenv = Environment.from_conda_specification(name="myenv", file_path="myenv.yml")
inference_config = InferenceConfig(entry_script="score.py", environment=myenv) 

# %%    
from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1, 
                                               tags = {'demo': 'onnx'}, 
                                               description = 'web service for MNIST ONNX model')
#%%
from azureml.core.model import Model
from random import randint

aci_service_name = 'onnx-demo-mnist'+str(randint(0,100))
print("Service", aci_service_name)
aci_service = Model.deploy(ws, aci_service_name, [model], inference_config, aciconfig)
aci_service.wait_for_deployment(True)
print(aci_service.state)

#%%

print(aci_service.scoring_uri)
