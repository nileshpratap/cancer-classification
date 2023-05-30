#%%
import shutil
from pathlib import Path
import pandas as pd
from utilities.confusionMatrix_dependent_functions import *

#%%
dataset_path = Path("../nnUNet_raw_data_base/nnUNet_raw_data")
data_path = dataset_path / "Task905_BrainCancerClassification"

#%%
rs_path = data_path / "testing"
seg_path = rs_path / "result"
gt_path = rs_path / "trail"
dict_yash_result = {}

calc_stats(gt_path, seg_path, dict_yash_result)

#%% dict to df
df_yash_result = pd.DataFrame.from_dict(dict_yash_result, orient='index')

#%%
df_yash_result.to_csv(str(data_path / "result_csv_file/yash_result") + ".csv")

