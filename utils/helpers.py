import os 
import importlib

def autoload_folder(context, callback):
    files = os.listdir(f"core/optimize/objectives/{context}") 
    names = [ file[:-3] for file in files if file.endswith(".py") ] 

    for name in names:
        module_name = (f"core.optimize.objectives.{context}.{name}")
        module = importlib.import_module(module_name) 
        callback(name, module) 