import pytest

from eemeter.caltrack.design_matrices import (
    create_caltrack_hourly_preliminary_design_matrix,
    create_caltrack_hourly_segmented_design_matrices,
    create_caltrack_daily_design_matrix,
    create_caltrack_billing_design_matrix,
)
from eemeter.features import estimate_hour_of_week_occupancy, fit_temperature_bins
from eemeter.segmentation import segment_time_series


def test_create_caltrack_hourly_preliminary_design_matrix(
    il_electricity_cdd_hdd_hourly
):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"]
    design_matrix = create_caltrack_hourly_preliminary_design_matrix(
        meter_data[:1000], temperature_data
    )
    assert design_matrix.shape == (1000, 7)
    assert sorted(design_matrix.columns) == [
        "cdd_65",
        "hdd_50",
        "hour_of_week",
        "meter_value",
        "n_hours_dropped",
        "n_hours_kept",
        "temperature_mean",
    ]
    assert round(design_matrix.sum().sum(), 2) == 136544.91


def test_create_caltrack_daily_design_matrix(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    design_matrix = create_caltrack_daily_design_matrix(
        meter_data[:100], temperature_data
    )
    assert design_matrix.shape == (100, 128)
    assert sorted(design_matrix.columns) == [
        "cdd_30",
        "cdd_31",
        "cdd_32",
        "cdd_33",
        "cdd_34",
        "cdd_35",
        "cdd_36",
        "cdd_37",
        "cdd_38",
        "cdd_39",
        "cdd_40",
        "cdd_41",
        "cdd_42",
        "cdd_43",
        "cdd_44",
        "cdd_45",
        "cdd_46",
        "cdd_47",
        "cdd_48",
        "cdd_49",
        "cdd_50",
        "cdd_51",
        "cdd_52",
        "cdd_53",
        "cdd_54",
        "cdd_55",
        "cdd_56",
        "cdd_57",
        "cdd_58",
        "cdd_59",
        "cdd_60",
        "cdd_61",
        "cdd_62",
        "cdd_63",
        "cdd_64",
        "cdd_65",
        "cdd_66",
        "cdd_67",
        "cdd_68",
        "cdd_69",
        "cdd_70",
        "cdd_71",
        "cdd_72",
        "cdd_73",
        "cdd_74",
        "cdd_75",
        "cdd_76",
        "cdd_77",
        "cdd_78",
        "cdd_79",
        "cdd_80",
        "cdd_81",
        "cdd_82",
        "cdd_83",
        "cdd_84",
        "cdd_85",
        "cdd_86",
        "cdd_87",
        "cdd_88",
        "cdd_89",
        "cdd_90",
        "hdd_30",
        "hdd_31",
        "hdd_32",
        "hdd_33",
        "hdd_34",
        "hdd_35",
        "hdd_36",
        "hdd_37",
        "hdd_38",
        "hdd_39",
        "hdd_40",
        "hdd_41",
        "hdd_42",
        "hdd_43",
        "hdd_44",
        "hdd_45",
        "hdd_46",
        "hdd_47",
        "hdd_48",
        "hdd_49",
        "hdd_50",
        "hdd_51",
        "hdd_52",
        "hdd_53",
        "hdd_54",
        "hdd_55",
        "hdd_56",
        "hdd_57",
        "hdd_58",
        "hdd_59",
        "hdd_60",
        "hdd_61",
        "hdd_62",
        "hdd_63",
        "hdd_64",
        "hdd_65",
        "hdd_66",
        "hdd_67",
        "hdd_68",
        "hdd_69",
        "hdd_70",
        "hdd_71",
        "hdd_72",
        "hdd_73",
        "hdd_74",
        "hdd_75",
        "hdd_76",
        "hdd_77",
        "hdd_78",
        "hdd_79",
        "hdd_80",
        "hdd_81",
        "hdd_82",
        "hdd_83",
        "hdd_84",
        "hdd_85",
        "hdd_86",
        "hdd_87",
        "hdd_88",
        "hdd_89",
        "hdd_90",
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
        "temperature_not_null",
        "temperature_null",
    ]
    assert round(design_matrix.sum().sum(), 2) == 167795.89


def test_create_caltrack_billing_design_matrix(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    design_matrix = create_caltrack_billing_design_matrix(
        meter_data[:10], temperature_data
    )
    assert design_matrix.shape == (10, 128)
    assert sorted(design_matrix.columns) == [
        "cdd_30",
        "cdd_31",
        "cdd_32",
        "cdd_33",
        "cdd_34",
        "cdd_35",
        "cdd_36",
        "cdd_37",
        "cdd_38",
        "cdd_39",
        "cdd_40",
        "cdd_41",
        "cdd_42",
        "cdd_43",
        "cdd_44",
        "cdd_45",
        "cdd_46",
        "cdd_47",
        "cdd_48",
        "cdd_49",
        "cdd_50",
        "cdd_51",
        "cdd_52",
        "cdd_53",
        "cdd_54",
        "cdd_55",
        "cdd_56",
        "cdd_57",
        "cdd_58",
        "cdd_59",
        "cdd_60",
        "cdd_61",
        "cdd_62",
        "cdd_63",
        "cdd_64",
        "cdd_65",
        "cdd_66",
        "cdd_67",
        "cdd_68",
        "cdd_69",
        "cdd_70",
        "cdd_71",
        "cdd_72",
        "cdd_73",
        "cdd_74",
        "cdd_75",
        "cdd_76",
        "cdd_77",
        "cdd_78",
        "cdd_79",
        "cdd_80",
        "cdd_81",
        "cdd_82",
        "cdd_83",
        "cdd_84",
        "cdd_85",
        "cdd_86",
        "cdd_87",
        "cdd_88",
        "cdd_89",
        "cdd_90",
        "hdd_30",
        "hdd_31",
        "hdd_32",
        "hdd_33",
        "hdd_34",
        "hdd_35",
        "hdd_36",
        "hdd_37",
        "hdd_38",
        "hdd_39",
        "hdd_40",
        "hdd_41",
        "hdd_42",
        "hdd_43",
        "hdd_44",
        "hdd_45",
        "hdd_46",
        "hdd_47",
        "hdd_48",
        "hdd_49",
        "hdd_50",
        "hdd_51",
        "hdd_52",
        "hdd_53",
        "hdd_54",
        "hdd_55",
        "hdd_56",
        "hdd_57",
        "hdd_58",
        "hdd_59",
        "hdd_60",
        "hdd_61",
        "hdd_62",
        "hdd_63",
        "hdd_64",
        "hdd_65",
        "hdd_66",
        "hdd_67",
        "hdd_68",
        "hdd_69",
        "hdd_70",
        "hdd_71",
        "hdd_72",
        "hdd_73",
        "hdd_74",
        "hdd_75",
        "hdd_76",
        "hdd_77",
        "hdd_78",
        "hdd_79",
        "hdd_80",
        "hdd_81",
        "hdd_82",
        "hdd_83",
        "hdd_84",
        "hdd_85",
        "hdd_86",
        "hdd_87",
        "hdd_88",
        "hdd_89",
        "hdd_90",
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
        "temperature_not_null",
        "temperature_null",
    ]
    assert round(design_matrix.sum().sum(), 2) == 19365.12


@pytest.fixture
def preliminary_hourly_design_matrix(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"]
    return create_caltrack_hourly_preliminary_design_matrix(
        meter_data[:1000], temperature_data
    )


@pytest.fixture
def segmentation(preliminary_hourly_design_matrix):
    return segment_time_series(
        preliminary_hourly_design_matrix.index, "three_month_weighted"
    )


@pytest.fixture
def occupancy_lookup(preliminary_hourly_design_matrix, segmentation):
    return estimate_hour_of_week_occupancy(
        preliminary_hourly_design_matrix, segmentation=segmentation
    )


@pytest.fixture
def temperature_bins(preliminary_hourly_design_matrix, segmentation):
    return fit_temperature_bins(
        preliminary_hourly_design_matrix, segmentation=segmentation
    )


def test_create_caltrack_hourly_segmented_design_matrices(
    preliminary_hourly_design_matrix, segmentation, occupancy_lookup, temperature_bins
):
    design_matrices = create_caltrack_hourly_segmented_design_matrices(
        preliminary_hourly_design_matrix,
        segmentation,
        occupancy_lookup,
        temperature_bins,
    )

    design_matrix = design_matrices["dec-jan-feb-weighted"]
    assert design_matrix.shape == (1000, 8)
    assert sorted(design_matrix.columns) == [
        "bin_0",
        "bin_1",
        "bin_2",
        "bin_3",
        "hour_of_week",
        "meter_value",
        "occupancy",
        "weight",
    ]
    assert round(design_matrix.sum().sum(), 2) == 126433.71

    design_matrix = design_matrices["mar-apr-may-weighted"]
    assert design_matrix.shape == (1000, 5)
    assert sorted(design_matrix.columns) == [
        "bin_0",
        "hour_of_week",
        "meter_value",
        "occupancy",
        "weight",
    ]
    assert round(design_matrix.sum().sum(), 2) == 0.0


def test_create_caltrack_billing_design_matrix_empty_temp(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"][:0]
    design_matrix = create_caltrack_billing_design_matrix(
        meter_data[:10], temperature_data
    )
    assert "n_days_kept" in design_matrix.columns


def test_create_caltrack_billing_design_matrix_partial_empty_temp(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"][:200]
    design_matrix = create_caltrack_billing_design_matrix(
        meter_data[:10], temperature_data
    )
    assert "n_days_kept" in design_matrix.columns
