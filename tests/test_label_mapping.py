from src.data.label_mapping import map_sleep_stage


def test_primary_label_mapping_excludes_preparation():
    assert map_sleep_stage("P") is None
    assert map_sleep_stage("Preparation") is None
    assert map_sleep_stage("W") == "Wake"
       assert map_sleep_stage("N1") == "Light"
    assert map_sleep_stage("N2") == "Light"
    assert map_sleep_stage("N3") == "Deep"
    assert map_sleep_stage("REM") == "REM"
