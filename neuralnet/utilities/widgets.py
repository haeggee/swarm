"""Python Script Template."""
import ipywidgets

# %% FLOAT WIDGETS
noise_widget = ipywidgets.FloatSlider(
    value=0.2, min=0, max=1, step=0.01, description='Noise:', continuous_update=False)

regularization_widget = ipywidgets.BoundedFloatText(
    value=0, min=0, max=1000, step=1e-4, description='Lambda:', continuous_update=False)

ratio_widget = ipywidgets.FloatLogSlider(
    value=1, min=-3, max=3, step=0.1, description='Ratio:', continuous_update=False)

bandwidth_widget = ipywidgets.FloatLogSlider(
    value=.1, min=-2, max=3, step=1e-4, description='Bandwidth', continuous_update=False
)

lr_widget = ipywidgets.FloatSlider(
    value=4, min=1e-1, max=10, step=1e-1, readout_format='.1f',
    description='Learning rate:', continuous_update=False)

min_prob_widget = ipywidgets.FloatSlider(
    value=.75, min=0, max=1, step=.01, description='Min Prob', continuous_update=False)


frequency_widget = ipywidgets.FloatSlider(
    value=0.1, min=0.1, max=2, step=1e-3, readout_format='.3f',
    description='Frequency:', continuous_update=False)

# %% INT WIDGETS
small_n_samples_widget = ipywidgets.IntSlider(
    value=30, min=10, max=300, step=1, description='Samples:', continuous_update=False)

big_n_samples_widget = ipywidgets.IntText(
    value=1500, step=50, description='Samples:', continuous_update=False)

n_features_widget = ipywidgets.IntSlider(
    value=10, min=1, max=10, description='Num Features:', continuous_update=False)

n_informative_widget = ipywidgets.IntSlider(
    value=1, min=1, max=10, description='Info Features:', continuous_update=False)

degree_widget = ipywidgets.IntSlider(
    value=1, min=1, max=9, step=1, description='Degree:', continuous_update=False)

code_size_widget = ipywidgets.IntSlider(
    value=16, min=1, max=64, description='Code Size', continuous_update=False)

n_iter_widget = ipywidgets.IntSlider(
    value=10, min=5, max=50, step=1, description='Num iter:', continuous_update=False)

n_centers_widget = ipywidgets.IntSlider(
    value=2, min=1, max=10, step=1,  description='Num Centers:', continuous_update=False
)

n_components_widget = ipywidgets.IntSlider(
    value=2, min=1, max=2,  description='Num Components:', continuous_update=False
)

batch_size_widget = ipywidgets.IntSlider(
    value=32, min=1, max=64, step=1, description='Batch Size:', continuous_update=False)

knn_widget = ipywidgets.IntSlider(
    value=1, min=1, max=9, step=1, description='k', continuous_update=False)


# %% STRING WIDGETS
CLASSIFICATION_DATASETS = [
    'blobs', 'circles', 'moons', 'anisotropic', 'varied variance', 'varied density',
    'no structure', 'xor', 'periodic',
    'linear', 'imbalanced', '2-blobs', '3-blobs', '4-blobs', 'iris']


KERNELS = ['rbf', 'poly', 'linear', 'laplacian', 'poly-2', 'poly-3']

classification_dataset_widget = ipywidgets.Dropdown(
    value='blobs', options=CLASSIFICATION_DATASETS,
    description='Dataset:', continuous_update=False)

kernel_widget = ipywidgets.Dropdown(
    value='rbf', options=KERNELS,
    description='Dataset:', continuous_update=False)

k_means_initialization_widget = ipywidgets.Dropdown(
    value='random', options=['random', 'random from data set', 'kmeans++'],
    description='Initialization:', continuous_update=False)

schedule_widget = ipywidgets.Dropdown(
    value='None', options=['Bold driver', 'AdaGrad', 'Annealing', 'None'],
    description='Learning rate heuristics:', continuous_update=False)

l1l2_widget = ipywidgets.Dropdown(
    value='l1', options=['l1', 'l2'], description='Regularization:',
    continuous_update=False)


# %% SELECT WIDGETS
plot_widget = ipywidgets.SelectMultiple(
    value=[], options=['Projected Data', 'Covariance Ellipse', 'Projection Line'],
    rows=4, description='Plot', disabled=False)

fold_widget = ipywidgets.ToggleButtons(
    value=1, options=[1, 2, 3, 4, 5],  description='Validation fold:',
    continuous_update=False)


# %% BUTTONS
resample_button = ipywidgets.ToggleButton(description="Resample!")
next_feature_button = ipywidgets.Button(description="Next Feature")
remove_feature_button = ipywidgets.Button(description="Remove Feature Feature")

assign_center_button = ipywidgets.Button(description="Assign Center")
update_mean_button = ipywidgets.Button(description="Update Means")

restart_button = ipywidgets.Button(description="Restart")
run_button = ipywidgets.Button(description="Run Algorithm")
