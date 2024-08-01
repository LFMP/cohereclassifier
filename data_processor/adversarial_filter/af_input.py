import os
import random

import datasets
from datasets import Dataset, load_from_disk
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class AFDataset:
  # Convert a loaded dataset in a suitable format for adversarial filtering process

  def __init__(self,
               dataset: Dataset,
               num_true_texts: int = 2,
               num_false_texts: int = 6,
               num_proc: int = 20):
    self.ds: Dataset = dataset
    self.num_proc = num_proc
    self.num_true_texts: int = num_true_texts
    self.num_false_texts: int = num_false_texts
    self.total_texts: int = num_true_texts * (self.num_false_texts + 1)
    self.prepare()

  def prepare(self) -> None:
    """
    This process is dataset-specific format. We spect the dataset has the
    following columns:
    - text_id: the unique identifier of the text (UUIDv4)
    - text: the original text
    - plot: the extracted plot from the original text
    - recreated_from: the id of the plot that was recreated from. id ==
      recreated_from means the text is not a recreation.
    - is_artificial: whether the text is generated or not
    - dataset: the dataset the text belongs to
    - manipulated_plot_n: the plot that was manipulated to generate the
      incoherent text
    - manipulation_type_n: the type of manipulation that was applied to the plot
    - text_n: the n-th manipulated text
    
    Output: a dataset with the following columns:
    - cluster_id: the unique identifier for the set of texts (UUIDv4)
    - text_0 to text_n: 1 of the n texts
    - plot_0 to plot_n: the plot of the text_0 to text_n
    - dataset: the dataset the texts belongs to
    - manipulation_0 to manipulation_n: the manipulation type of the text_0 to text_n
    - text_id_0 to text_id_n: the id of the text_0 to text_n
    - label_0 label_n: the label of the text_0 to text_n
    """
    # Create a new dataset

    new_dataset: dict[str, list] = {
        'cluster_id': [],
        'dataset': [],
    }
    var_columns: list[str] = [
        'text_', 'plot_', 'is_artificial_', 'manipulation_', 'label_'
    ]
    new_dataset.update({
        f'{var}{i}': [] for var in var_columns for i in range(self.total_texts)
    })

    # get the UUIDs from the original dataset to use as cluster_id
    unique_clusters = self.ds.unique('recreated_from')
    # split the dataset into clusters by the recreated_from column and
    # unique_cluster
    datasets.disable_progress_bar()
    for cluster in tqdm(unique_clusters, desc='Processing clusters'):
      cluster_data: Dataset = self.ds.filter(
          lambda x: x['recreated_from'] == cluster, num_proc=self.num_proc)
      # common from all texts in the cluster
      new_dataset['cluster_id'].append(cluster)
      new_dataset['dataset'].append(cluster_data[0]['dataset'])
      used_idx = []
      possible_idx = range(self.total_texts)
      for cluster_row_idx in range(self.num_true_texts):
        row_data = cluster_data[cluster_row_idx]
        idx: int = random.choice([i for i in possible_idx if i not in used_idx])
        new_dataset[f'text_{idx}'].append(row_data['text'])
        new_dataset[f'plot_{idx}'].append(row_data['plot'])
        new_dataset[f'is_artificial_{idx}'].append(row_data['is_artificial'])
        new_dataset[f'manipulation_{idx}'].append([])
        new_dataset[f'label_{idx}'].append(1)
        used_idx.append(idx)
        for text_idx in range(self.num_false_texts):
          idx: int = random.choice(
              [i for i in possible_idx if i not in used_idx])
          new_dataset[f'text_{idx}'].append(row_data[f'text_{text_idx}'])
          new_dataset[f'plot_{idx}'].append(
              row_data[f'manipulated_plot_{text_idx}'])
          new_dataset[f'manipulation_{idx}'].append(
              row_data[f'manipulation_type_{text_idx}'])
          new_dataset[f'label_{idx}'].append(0)
          new_dataset[f'is_artificial_{idx}'].append(True)
          used_idx.append(idx)

    # change the current dataset to the new one
    self.ds = Dataset.from_dict(new_dataset)
    self.ds = self.ds.class_encode_column('dataset')
    for idx in range(self.total_texts):
      self.ds = self.ds.class_encode_column(f"label_{idx}")