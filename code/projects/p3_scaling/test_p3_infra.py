"""Tests for the Project 3 infrastructure (training service + client + ledger). No network."""
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ledger import HTTPClient, Ledger  # noqa: E402
from training_service import serve  # noqa: E402


def body(layers=4, hidden=256, tokens=512 * 64 * 200, lr=3e-3, max_rt=3600.0):
    return {"architecture_config": {"hidden_size": hidden, "num_attention_heads": hidden // 64,
                                    "head_dim": 64, "num_key_value_heads": hidden // 64,
                                    "num_hidden_layers": layers, "intermediate_size": 704,
                                    "vocab_size": 32000, "tie_word_embeddings": False},
            "optimizer_config": {"lr_scheduler": {"peak_value": lr}},
            "train_batch_size": 64, "total_train_tokens": tokens, "n_evals": 4,
            "max_runtime_seconds": max_rt, "model_seed": 0}


@pytest.fixture()
def service():
    srv, svc = serve(0, budget_hours=2.0, time_scale=1e6, api_key="00000000")
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", svc
    srv.shutdown()


def test_submit_poll_ledger(service, tmp_path):
    url, svc = service
    client = HTTPClient(url, "00000000", backoff=0.01)
    ledger = Ledger(str(tmp_path / "ledger.jsonl"))
    ids = [ledger.submit_once(client, body(layers=l), tag=f"L{l}") for l in (2, 4)]
    # at submission time the full max_runtime is reserved against the budget
    first = next(iter(ledger.submitted.values()))
    assert first["budget"]["used_seconds"] == pytest.approx(3600)
    # resubmitting through the ledger is free and returns the same id
    assert ledger.submit_once(client, body(layers=2)) == ids[0]
    # resubmitting directly is rejected by the service as a duplicate
    with pytest.raises(RuntimeError, match="409"):
        client.submit(body(layers=2))
    res = ledger.poll(client, ids, interval=0.01)
    assert {r["status"]["status_type"] for r in res} == {"completed"}
    used = client.budget()["used_seconds"]
    assert 0 < used < 2 * 3600                                              # refunded
    # a fresh Ledger object rebuilds state from disk
    again = Ledger(str(tmp_path / "ledger.jsonl"))
    rows = again.table()
    assert len(rows) == 2 and all(r["final_loss"] is not None for r in rows)


def test_budget_and_auth_errors(service):
    url, _ = service
    with pytest.raises(RuntimeError, match="401"):
        HTTPClient(url, "bad", backoff=0.01).budget()
    client = HTTPClient(url, "00000000", backoff=0.01)
    with pytest.raises(RuntimeError, match="400"):
        client.submit(body(max_rt=3 * 3600))                                # > 2 h budget
    with pytest.raises(RuntimeError, match="422"):
        client.submit(body(tokens=12345))                                   # not whole batches


def test_timeout_is_charged_in_full(service):
    url, _ = service
    client = HTTPClient(url, "00000000", backoff=0.01)
    eid = client.submit(body(layers=8, hidden=512, tokens=512 * 64 * 4000, max_rt=60.0))["experiment_id"]
    import time
    for _ in range(200):
        e = client.experiment(eid)
        if e["status"]["status_type"] in ("completed", "failed"):
            break
        time.sleep(0.01)
    assert e["status"]["status_type"] == "failed"
    assert e["status"]["reason"]["reason"] == "timeout"
    assert e["status"]["used_runtime_seconds"] == pytest.approx(60.0)


def test_score_and_randomised_truth(service):
    url, svc = service
    client = HTTPClient(url, "00000000", backoff=0.01)
    with pytest.raises(RuntimeError, match="400"):
        client._call("GET", "/score?hours=2")
    client.final_submission({"training_config": body(), "predicted_final_loss": 3.0})
    sc = client._call("GET", "/score?hours=2")
    assert sc["regret"] >= 0 and "oracle_loss" in sc
    from llmsys.scaling import GroundTruth
    a, b = GroundTruth.sample(1), GroundTruth.sample(2)
    assert a != b and a == GroundTruth.sample(1)
