import numpy as np
import pandas as pd
import pytest

from cheese_fate.combase import batch_id, load_exports, source_id
from cheese_fate.curation import (
    align,
    audit,
    build_observations,
    flag_censoring,
    ph_rebound,
)


def write_export(path, records, logcs, temperatures=None, phs=None):
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(records).to_excel(writer, sheet_name="Data Records", index=False)
        pd.DataFrame(logcs, columns=["Record ID", "Time", "Logc"]).to_excel(
            writer, sheet_name="Logcs", index=False
        )
        if temperatures is not None:
            pd.DataFrame(temperatures, columns=["Record ID", "Time", "Temperature"]).to_excel(
                writer, sheet_name="Temperatures", index=False
            )
        if phs is not None:
            pd.DataFrame(phs, columns=["Record ID", "Time", "pH"]).to_excel(
                writer, sheet_name="pHs", index=False
            )


def record(record_id, organism, conditions="Raw"):
    return {
        "Record ID": record_id,
        "Organism": organism,
        "Food Name": "test cheese",
        "Conditions": conditions,
    }


@pytest.fixture
def export(tmp_path):
    ecoli = [("Bat1_2", t, c) for t, c in [(0, 3.0), (10, 4.0), (20, 3.5), (30, 3.0), (40, 2.5)]]
    lab = [("Bat1_10", t, c) for t, c in [(0, 6.0), (20, 8.0), (40, 8.5)]]
    lab_other = [("Bat1_11", 20, 8.4)]
    write_export(
        tmp_path / "export_001.xlsx",
        [record("Bat1_2", "Escherichia coli"), record("Bat1_10", "Lactic acid bacteria")],
        ecoli + lab,
        temperatures=[("Bat1_2", 0, 37), ("Bat1_2", 2, 50), ("Bat1_2", 5, 12)],
        phs=[("Bat1_2", 0, 6.6), ("Bat1_2", 20, 5.0), ("Bat1_2", 40, 5.6)],
    )
    # Second file without condition sheets; repeats one record, as ComBase exports can.
    write_export(
        tmp_path / "export_002.xlsx",
        [record("Bat1_11", "Lactic acid bacteria", ""), record("Bat1_2", "Escherichia coli")],
        lab_other + ecoli,
    )
    return load_exports(tmp_path)


def test_static_conditions_fill_gaps_and_implausible_values_are_dropped(tmp_path):
    rec = {**record("Iso_1", "Escherichia coli"), "Temperature (C)": 12.0, "pH": 5.4, "Aw": 0.24}
    dyn = {**record("Dyn_1", "Escherichia coli"), "Temperature (C)": "0;37;5;12", "pH": 6.0}
    counts = [("Iso_1", t, 3.0) for t in (0, 24, 48, 72, 96)]
    counts += [("Dyn_1", t, 3.0) for t in (0, 24, 48, 72, 96)]
    write_export(
        tmp_path / "e.xlsx", [rec, dyn], counts, temperatures=[("Dyn_1", 0, 37), ("Dyn_1", 5, 12)]
    )
    export = load_exports(tmp_path)
    cond = export.conditions.groupby(["record_id", "variable"])
    origins = cond["origin"].agg(set).to_dict()
    values = cond["value"].agg(list).to_dict()
    assert origins[("Iso_1", "temperature_c")] == {"static"}
    assert ("Iso_1", "aw") not in origins  # 0.24 is not a plausible aw
    # The series wins over the record-sheet string; pH comes from the record sheet.
    assert origins[("Dyn_1", "temperature_c")] == {"series"}
    assert values[("Dyn_1", "ph")] == [6.0]

    obs = build_observations(export, ["Escherichia coli"])
    iso = obs[obs["record_id"] == "Iso_1"]
    assert iso["temperature_c"].eq(12.0).all() and iso["ph"].eq(5.4).all()
    assert iso["temperature_origin"].eq("static").all()
    assert obs.loc[obs["record_id"] == "Dyn_1", "temperature_origin"].eq("series").all()


def test_record_id_parsing():
    assert batch_id("BitBo1_10") == "BitBo1"
    assert batch_id("cavpu-FG-4_1_3") == "cavpu-FG-4_1"
    assert batch_id("Lm2a") == "Lm2a"
    assert source_id("cavpu-FG-4_1_3") == "cavpu"
    assert source_id("BitBo1_3") == "BitBo"


def test_load_exports_tolerates_missing_sheets_and_duplicates(export):
    assert sorted(export.records["record_id"]) == ["Bat1_10", "Bat1_11", "Bat1_2"]
    assert export.counts.duplicated().sum() == 0
    assert (export.counts["record_id"] == "Bat1_2").sum() == 5
    assert set(export.conditions["variable"]) == {"temperature_c", "ph"}
    assert export.records.set_index("record_id").loc["Bat1_2", "raw_milk"]
    assert not export.records.set_index("record_id").loc["Bat1_11", "raw_milk"]


def test_align_step_and_linear():
    times = np.array([-1.0, 0.0, 3.0, 10.0])
    values, flags = align(times, np.array([0.0, 2.0, 5.0]), np.array([37, 50, 12]), "step")
    assert np.isnan(values[0])
    assert values[1:].tolist() == [37, 50, 12]
    assert flags.tolist() == [False, False, True, True]

    values, flags = align(times, np.array([0.0, 10.0]), np.array([6.0, 5.0]), "linear")
    assert np.isnan(values[0])
    assert values[2] == pytest.approx(5.7)
    assert flags.tolist() == [False, False, True, False]


def test_build_observations_links_same_batch_lab(export):
    obs = build_observations(export, ["Escherichia coli"], min_points=5)

    assert obs["record_id"].unique().tolist() == ["Bat1_2"]
    assert obs["time_h"].tolist() == [0, 10, 20, 30, 40]
    # Counts are never interpolated.
    assert obs["log_count"].tolist() == [3.0, 4.0, 3.5, 3.0, 2.5]
    assert obs["temperature_c"].tolist() == [37, 12, 12, 12, 12]
    assert obs["ph"].iloc[1] == pytest.approx(5.8)
    assert obs["ph_interpolated"].tolist() == [False, True, False, True, False]
    # LAB at t=20 is the max over both LAB records; t=10 is interpolated.
    assert obs["lab_log_count"].tolist() == pytest.approx([6.0, 7.2, 8.4, 8.45, 8.5])
    assert obs["lab_log_count_interpolated"].tolist() == [False, True, False, True, False]
    assert obs["aw"].isna().all()


def test_build_observations_applies_min_points(export):
    with pytest.raises(ValueError, match="at least 6"):
        build_observations(export, ["Escherichia coli"], min_points=6)


def test_audit_counts_dynamic_conditions(export):
    summary = audit(export, min_points=5)
    row = summary.loc["Escherichia coli"]
    assert row["curves"] == 1
    assert row["temperature_varies"] == 1
    assert row["ph_varies"] == 1
    assert row["lab_same_batch"] == 1
    assert "Lactic acid bacteria" not in summary.index


def test_flag_censoring_uses_nd_symbol_and_source_detection_limit():
    obs = pd.DataFrame(
        {
            "source_id": ["s1", "s1", "s1", "s2"],
            "organism": ["Escherichia coli"] * 4,
            "log_count": [2.3, -0.01, 1.48, 0.01],
        }
    )
    flagged = flag_censoring(obs)
    assert flagged["censored"].tolist() == [False, True, False, True]
    assert flagged["log_count_raw"].tolist() == [2.3, -0.01, 1.48, 0.01]
    # s1 limit = smallest detected count; s2 has none and gets the default.
    assert flagged["detection_limit"].tolist() == [1.48, 1.48, 1.48, 1.0]
    assert flagged["detection_limit_imputed"].tolist() == [False, False, False, True]
    assert flagged["log_count"].tolist() == [2.3, 1.48, 1.48, 1.0]


def test_ph_rebound():
    assert ph_rebound([0, 10, 20, 30], [6.6, 5.0, 5.4, 5.3]) == (10.0, pytest.approx(0.4))
    assert np.isnan(ph_rebound([0], [6.0])[1])
