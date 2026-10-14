def test_smooth_table_written(tmp_path, monkeypatch):
    from common import paths
    monkeypatch.setattr(paths, "ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("PATE_RESULTS_DIR", str(tmp_path / "results"))
    import pandas as pd
    from member2_privacy import dev_sweep, smooth_table
    dev_sweep.run(teachers=[100, 250], scales=[20], queries=[100, 500], seeds=[42, 43], n_public=1000)
    smooth_table.smooth_table()
    t = pd.read_csv(tmp_path / "results" / "smooth_sensitivity_table.csv")
    assert len(t) == 4 and (t.n_seeds == 2).all() and (t.release_noise_max >= 0).all()
