import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder



def GRS_pic(GRS, model_name, save_path=None):
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
    matplotlib.axes.Axes
        Axes object of the GRS distribution plot.
    """
    #计算m1,m2
    disease_types = GRS['Disease'].unique()  # 获取Disease中所有唯一值
    m1 = np.mean(GRS[GRS['Disease'] == disease_types[0]]['grs'])
    m2 = np.mean(GRS[GRS['Disease'] == disease_types[1]]['grs'])
    # 绘制图形
    plt.figure(figsize=(10, 6))
    pg = sns.kdeplot(data=GRS, x='grs', hue='Disease', fill=True, common_norm=False, alpha=0.8, palette=['#2166AC', '#B2182B'])

    # 添加垂直线
    plt.axvline(x=m1, linestyle='dashed', color='#2166AC')
    plt.axvline(x=m2, linestyle='dashed', color='#B2182B')

    # 设置主题
    sns.set_theme(style="whitegrid")
    plt.grid(False)

    # 设置填充颜色
    handles, labels = pg.get_legend_handles_labels()
    plt.legend(handles, labels, title='')

    # 设置标签
    plt.xlabel('Genetic Risk Score')
    plt.ylabel('Percentage')
    
    # 保存图形
    if save_path and model_name:
        plt.savefig(f'{save_path}/{model_name}_grs.png')
    plt.show()
    plt.close()  # 关闭当前图形，防止重叠

    return pg


def ROC_pic(GRS, model_name, save_path=None):
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
    matplotlib.figure.Figure
        Figure object of the ROC curve plot.
    """

    # 提取标签和预测值
    label_encoder = LabelEncoder()# 将Disease列转换为数值标签,这个转化方法可以看看看
    y_true = label_encoder.fit_transform(GRS['Disease'])
    y_scores = GRS['grs']

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    pg=plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # 添加标签和标题
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    # 显示AUC值
    plt.text(0.6, 0.5, f'AUC = {roc_auc:.3f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

    # 显示和保存图形
    if save_path and model_name:
        plt.savefig(f'{save_path}/{model_name}_roc.png')
    plt.show()
    plt.close()  # 关闭当前图形，防止重叠

    return pg


if __name__ == '__main__':
    # 构造模拟数据
    np.random.seed(42)
    n_samples = 100
    GRS = pd.DataFrame({
        'grs': np.random.normal(loc=0, scale=1, size=n_samples),
        'Disease': np.random.choice(['Healthy', 'Disease'], size=n_samples)
    })

    # 测试 GRS_pic
    print("Testing GRS_pic...")
    GRS_pic(GRS, model_name='test_model', save_path=None)

    # 测试 ROC_pic
    print("Testing ROC_pic...")
    ROC_pic(GRS, model_name='test_model', save_path=None)