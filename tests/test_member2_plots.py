def test_plots_and_table(tmp_path, monkeypatch):
    from common import paths
    monkeypatch.setattr(paths, "ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("PATE_RESULTS_DIR", str(tmp_path / "results"))
    import pandas as pd
    from member2_privacy import dev_sweep, plots
    dev_sweep.run(teachers=[10, 100], scales=[5, 20], queries=[100, 500], seeds=[42, 43], n_public=1000)
    out = plots.main(n_queries=500, ref_teachers=100, ref_scale=20)
    assert len(out) == 5 and all(p.exists() and p.stat().st_size > 5000 for p in out)   # fig5 cost vs gap, fig6 teacher trade-off
    df = pd.read_csv(tmp_path / "results" / "epsilon_table.csv")
    assert len(df) == 2 * 2 * 2 and (df.n_seeds == 2).all()
