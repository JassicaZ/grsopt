import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder



def grs_pic(GRS, model_name, save_path=None):
    """
    Draw the distribution curve for Genetic Risk Score (GRS).

    Parameters
    ----------
    GRS : pd.DataFrame
        DataFrame containing 'grs' (Genetic Risk Score) and 'Disease' columns.
    model_name : str
        Name of the model or method used for GRS calculation/ optimization. Used for saving the figure.
    save_path : str, optional
        Directory path to save the generated plot.

    Returns
    -------
    None
        The figure is displayed (and saved when ``save_path`` is given). It is
        deliberately not returned: in Jupyter the returned artist would be
        rendered a second time as the cell result, duplicating the plot.
    """
    # Compute the GRS for the two groups.
    disease_types = GRS['Disease'].unique()
    m1 = np.mean(GRS[GRS['Disease'] == disease_types[0]]['grs'])
    m2 = np.mean(GRS[GRS['Disease'] == disease_types[1]]['grs'])

    # Plot the GRS density distribution by disease status.
    fig = plt.figure(figsize=(10, 6))
    pg = sns.kdeplot(data=GRS, x='grs', hue='Disease', fill=True, common_norm=False, alpha=0.8, palette=['#2166AC', '#B2182B'])

    plt.axvline(x=m1, linestyle='dashed', color='#2166AC')
    plt.axvline(x=m2, linestyle='dashed', color='#B2182B')

    sns.set_theme(style="whitegrid")
    plt.grid(False)

    handles, labels = pg.get_legend_handles_labels()
    plt.legend(handles, labels, title='')

    plt.xlabel('Genetic Risk Score')
    plt.ylabel('Percentage')

    # Save the figure if both the output directory and model name are provided.
    if save_path and model_name:
        plt.savefig(f'{save_path}/{model_name}_grs.png')
    plt.show()
    plt.close(fig)  # Close the figure to avoid overlap between plots.


def roc_pic(GRS, model_name, save_path=None):
    """
    Plot the ROC curve for Genetic Risk Score (GRS).

    Parameters
    ----------
    GRS : pd.DataFrame
        DataFrame containing 'grs' (Genetic Risk Score) and 'Disease' columns.
    model_name : str
        Name of the model or method used for GRS calculation/ optimization. Used for saving the figure.
    save_path : str, optional
        Directory path to save the generated plot.

    Returns
    -------
    None
        The figure is displayed (and saved when ``save_path`` is given). It is
        deliberately not returned: in Jupyter the returned figure would be
        rendered a second time as the cell result, duplicating the plot.
    """

    # Encode disease labels to numeric values for ROC analysis.
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(GRS['Disease'])
    y_scores = GRS['grs']

    # Compute the ROC curve and AUC value.
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve.
    pg = plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    # Display the AUC score on the plot.
    plt.text(0.6, 0.5, f'AUC = {roc_auc:.3f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

    # Display and save the figure if the output path and model name are provided.
    if save_path and model_name:
        plt.savefig(f'{save_path}/{model_name}_roc.png')
    plt.show()
    plt.close(pg)  # Close the figure to avoid overlap between plots.


if __name__ == '__main__':
    # Generate synthetic data for testing purposes.
    np.random.seed(42)
    n_samples = 100
    GRS = pd.DataFrame({
        'grs': np.random.normal(loc=0, scale=1, size=n_samples),
        'Disease': np.random.choice(['Healthy', 'Disease'], size=n_samples)
    })

    # Test the GRS distribution plot.
    print("Testing grs_pic...")
    grs_pic(GRS, model_name='test_model', save_path=None)

    # Test the ROC plot.
    print("Testing roc_pic...")
    roc_pic(GRS, model_name='test_model', save_path=None)