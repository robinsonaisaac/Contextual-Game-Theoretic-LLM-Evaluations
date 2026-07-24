import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_tier5_ledger import build_traces


def test_dryrun_composition_matches_spec():
    stats = build_traces(dry_run=True)
    assert stats["n_long"] == 2560                # 4 families x 640
    assert abs(stats["frac_recovery"] - 0.12) < 0.03
    assert abs(stats["frac_short_noledger"] - 0.15) < 0.03
    assert stats["n_indomain_eval_sets"] >= 4     # >=1 extrapolated set per train family
    assert stats["n_heldout_each"] == 100
    # every training completion ends with an ANSWER line and every eval carries a gold answer
    assert stats["all_train_have_answer_tail"] is True
    assert stats["all_eval_have_gold"] is True
