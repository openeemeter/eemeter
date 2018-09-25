from eemeter.warnings import EEMeterWarning


def test_eemeter_warning():
    eemeter_warning = EEMeterWarning(
        qualified_name="qualified_name", description="description", data={}
    )
    assert eemeter_warning.qualified_name == "qualified_name"
    assert eemeter_warning.description == "description"
    assert eemeter_warning.data == {}
    assert str(eemeter_warning).startswith("EEMeterWarning")
    assert eemeter_warning.json() == {
        "data": {},
        "description": "description",
        "qualified_name": "qualified_name",
    }
