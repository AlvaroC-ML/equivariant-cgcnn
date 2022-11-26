## GVP Model

### This code creates a GNN architecture I designed which uses Geometric Vector Perceptrons (GVPs).

The reason why I use GVPs is because of their rotational equivariance. For more details about why this is a desired quality, please wait until I have finished my paper explaining the math behind this. As a SULI intern, I have to write this paper. It will contain all the details about what exactly this architecture does.

- layers_gvp.py: Contains the layers used in the model.
- model_gvp.py: Contains the model.
- SpektralDataset_gvp.py: Downloads data and prepares it to be used by Spektral.
- preprocessor_gvp.py: Reads a cif file of a crystal and creates the corresponding crystal graph.
- train_gvp.py: Trains and evaluates model.
- rot_eq_test.py: Verifies that the model is rotationally equivariant.
