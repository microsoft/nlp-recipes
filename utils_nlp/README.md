# NLP Utilities

This module (**utils_nlp**) contains functions to simplify common tasks used when developing and evaluating NLP systems. For more details about what functions are available and how to use them, please review the doc-strings provided with the code.

## Submodules
The utilities package is made up of several high-level submodules for common utilies, datasets, evaluation and model interpretability. It also includes code that make it easy to interact with various platforms and frameworks.  

For more information about the individual submodules, find links below:  

- [Azure Machine Learning](./azureml/README.md) - Contains Azure Machine Learning specific utilities  
- [Common](./common/README.md) - Contains common helper utilities such as  
        - `Timer`: A timer object that helps with timing execution runs.  
        - `get_device` and `move_to_device`: Pytorch specific utilities that help determine the compute device and handle moving of models across various types of compute respectively. 
- [Dataset](./dataset/README.md) - Contains dataset definition and sources  
- [Evaluation (Eval)](./eval/README.md) - Contains metric and accuracy evaluation utilities     
- [Models](./models/README.md) - Contains implementation of algorithms used     
- [Interpreter](./interpreter/README.md) - Contains utilities to explain hidden states of models i.e. **model interpretability**. 

