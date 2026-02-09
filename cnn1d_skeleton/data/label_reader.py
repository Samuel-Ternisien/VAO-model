from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List, Tuple

def read_events_segments(events_csv : str | Path) -> List[Tuple[int, int, int]]:
    """
    Read Events file:
    Name;Class;Frame Start;Timestamp Start;Frame End;Timestamp End
    Give [(frame_start, frame_end, class_id), ...]
    """

    df = pd.read_csv(events_csv, sep=";")

    fs = df["Frame Start"].astype(int).to_list()
    fe = df["Frame End"].astype(int).to_list()
    cl = df["Class"].astype(int).to_list()
    return list(zip(fs, fe, cl))