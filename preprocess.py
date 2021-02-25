# %%
import pickle
from pathlib import Path

# %%
p = Path("./PPG_FieldStudy/S1/S1.pkl")
file = open(p, 'rb')
with p.open() as f:
    # show = pickle.load(f)
    f.read



# %%
