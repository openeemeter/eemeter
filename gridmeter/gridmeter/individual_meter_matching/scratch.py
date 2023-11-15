from __future__ import annotations

import pandas as pd

from gridmeter.individual_meter_matching.create_comparison_groups import Individual_Meter_Matching as IMM
from gridmeter.individual_meter_matching import settings as _settings

def _main():
    df_ls_t = pd.read_csv("/app/gridmeter/data/df_ls_t.csv")
    df_ls_cp = pd.read_csv("/app/gridmeter/data/df_ls_cp.csv")
    s = _settings.Settings()
    res = IMM(s).get_full_cg(df_ls_t, df_ls_cp)
    print(res)


if __name__ == "__main__":
    _main()