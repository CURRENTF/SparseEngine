import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.long_bench import pred as longbench_pred
from benchmark.long_bench.metrics import classification_score, qa_f1_score
from benchmark.sparseengine_regression.manifest import load_manifest, resolve_method_config


class LongBenchDeltaKVContractsTest(unittest.TestCase):
    def _omnikv_args(self, hyper_param):
        return SimpleNamespace(
            hyper_param=json.dumps(hyper_param),
            sparse_method="omnikv",
            max_model_len=32768,
            deltakv_checkpoint_path=None,
            allow_single_omnikv_full_layer=False,
        )

    def test_longbench_requires_explicit_prefix_caching_opt_in(self):
        args = self._omnikv_args(
            resolve_method_config(
                load_manifest()["methods"]["omnikv"],
                model_id="qwen25_7b",
                require_model_config=True,
            )
        )
        config = longbench_pred._build_infer_config(args)
        self.assertIs(config["enable_prefix_caching"], False)

        requested = json.loads(args.hyper_param)
        requested["enable_prefix_caching"] = True
        args.hyper_param = json.dumps(requested)
        with self.assertRaisesRegex(ValueError, "enable_prefix_caching=False"):
            longbench_pred._build_infer_config(args)

        args.allow_prefix_caching = True
        config = longbench_pred._build_infer_config(args)
        self.assertIs(config["enable_prefix_caching"], True)

    def test_longbench_records_requested_and_effective_omnikv_config(self):
        config = resolve_method_config(
            load_manifest()["methods"]["omnikv"],
            model_id="qwen25_7b",
            require_model_config=True,
        )
        args = self._omnikv_args(config)
        infer_config = longbench_pred._build_infer_config(args)
        requested = longbench_pred._requested_runtime_config(args, infer_config)
        requested_layers = requested["config"]["full_attention_layers"]
        self.assertTrue(requested_layers)

        runtime_info = {
            "sparse_method": "omnikv",
            "full_attention_layers": requested_layers,
        }
        generate_fn = SimpleNamespace(
            _sparseengine_llm=SimpleNamespace(
                worker_info=lambda **_kwargs: runtime_info,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            resolved = Path(tmp) / "resolved_config.json"
            resolved.write_text(
                json.dumps({"backend": "sparseengine", "requested": requested}),
                encoding="utf-8",
            )
            longbench_pred._record_effective_runtime_config(
                generate_fn=generate_fn,
                out_root=tmp,
            )
            recorded = json.loads(resolved.read_text(encoding="utf-8"))

        self.assertEqual(recorded["requested"], requested)
        self.assertEqual(recorded["effective_runtime"], runtime_info)
        self.assertEqual(
            recorded["requested"]["config"]["full_attention_layers"],
            recorded["effective_runtime"]["full_attention_layers"],
        )

    def test_longbench_records_final_prefix_cache_statistics(self):
        generate_fn = SimpleNamespace(
            _sparseengine_llm=SimpleNamespace(
                worker_load=lambda: {
                    "active_requests": 0,
                    "cache": {
                        "prefix_cache_hit_requests": 3,
                        "prefix_cache_hit_tokens": 48,
                    },
                },
                debug_sparse_state_summaries=lambda synchronize: [
                    {"world_rank": 0, "state": {"dynamic_selection": {"rkv_compactions": 2}}}
                ],
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            longbench_pred._write_worker_load_stats(
                generate_fn=generate_fn,
                out_root=tmp,
                rank=0,
            )
            recorded = json.loads(
                (Path(tmp) / "worker_load_stats_rank0.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            recorded["worker_load"]["cache"]["prefix_cache_hit_tokens"],
            48,
        )
        self.assertEqual(
            recorded["sparse_state"][0]["state"]["dynamic_selection"]["rkv_compactions"],
            2,
        )

    def test_chat_template_policy_matches_regular_prompt_paths(self):
        self.assertTrue(
            longbench_pred.should_use_chat_template("hotpotqa", thinking_mode="off")
        )
        self.assertTrue(
            longbench_pred.should_use_chat_template("hotpotqa", thinking_mode="on_strip")
        )
        self.assertFalse(
            longbench_pred.should_use_chat_template("hotpotqa", no_chat_template=True)
        )

    def test_hotpotqa_and_trec_metric_contracts(self):
        self.assertEqual(qa_f1_score("Paris", "Paris"), 1.0)
        self.assertEqual(qa_f1_score("Paris", "London"), 0)
        self.assertEqual(
            classification_score(
                "DESC",
                "DESC",
                all_classes=["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"],
            ),
            1.0,
        )

    def test_longbench_data_validation_fails_fast_for_missing_hotpotqa_and_trec(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_root = longbench_pred.DATA_PREFIX_PATH
            longbench_pred.DATA_PREFIX_PATH = str(Path(tmp) / "missing")
            try:
                with self.assertRaisesRegex(FileNotFoundError, "LongBench data root"):
                    longbench_pred.validate_longbench_data_paths(["hotpotqa", "trec"], use_longbench_e=False)
            finally:
                longbench_pred.DATA_PREFIX_PATH = old_root

    def test_longbench_data_validation_requires_explicit_root(self):
        old_root = longbench_pred.DATA_PREFIX_PATH
        longbench_pred.DATA_PREFIX_PATH = None
        try:
            with self.assertRaisesRegex(FileNotFoundError, "SPARSEENGINE_LONGBENCH_DATA_DIR"):
                longbench_pred.validate_longbench_data_paths(["hotpotqa"], use_longbench_e=False)
        finally:
            longbench_pred.DATA_PREFIX_PATH = old_root

    def test_sparseengine_data_workers_receive_distinct_master_ports(self):
        launched = []

        class Process:
            def wait(self):
                return 0

        def fake_popen(command, *, env, cwd):
            launched.append((command, env, cwd))
            return Process()

        worker_args = SimpleNamespace(ws=4)
        with (
            patch.dict(
                "os.environ",
                {
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                    "SPARSEENGINE_MASTER_PORT": "24300",
                },
                clear=False,
            ),
            patch.object(longbench_pred.subprocess, "Popen", side_effect=fake_popen),
        ):
            longbench_pred.launch_single_gpu_workers(worker_args, "/tmp/longbench-output")

        self.assertEqual(
            [env["CUDA_VISIBLE_DEVICES"] for _command, env, _cwd in launched],
            ["0", "1", "2", "3"],
        )
        self.assertEqual(
            [env["SPARSEENGINE_MASTER_PORT"] for _command, env, _cwd in launched],
            ["24300", "24301", "24302", "24303"],
        )

    def test_longbench_records_actual_decode_cuda_graph_state(self):
        graph_runner = SimpleNamespace(
            _graphs={
                "captured": SimpleNamespace(graph=object()),
                "uncaptured": SimpleNamespace(graph=None),
            },
            last_state_key="captured",
            capture_count=2,
            replay_count=7,
            eager_static_count=0,
            force_eager_count=0,
        )
        generate_fn = SimpleNamespace(
            _sparseengine_llm=SimpleNamespace(
                config=SimpleNamespace(decode_graph=True),
                model_runner=SimpleNamespace(
                    decode_graph_runner=graph_runner,
                ),
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            status = longbench_pred._write_decode_cuda_graph_status(
                generate_fn=generate_fn,
                out_root=tmp,
                rank=2,
            )
            path = Path(tmp) / "decode_graph_status_rank2.json"

            self.assertEqual(status["rank"], 2)
            self.assertTrue(status["configured"])
            self.assertTrue(status["runner_initialized"])
            self.assertEqual(status["state_count"], 2)
            self.assertEqual(status["graph_count"], 1)
            self.assertTrue(status["active"])
            self.assertEqual(status["capture_count"], 2)
            self.assertEqual(status["replay_count"], 7)
            self.assertEqual(status["last_state_key"], "captured")
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                status,
            )

    def test_longbench_records_business_graph_counter_delta(self):
        graph_runner = SimpleNamespace(
            _graphs={"captured": SimpleNamespace(graph=object())},
            last_state_key="captured",
            capture_count=1,
            replay_count=3,
            eager_static_count=0,
            force_eager_count=0,
        )
        generate_fn = SimpleNamespace(
            _sparseengine_llm=SimpleNamespace(
                config=SimpleNamespace(decode_graph=True),
                model_runner=SimpleNamespace(
                    decode_graph_runner=graph_runner,
                ),
            )
        )
        before = longbench_pred._decode_cuda_graph_status(
            generate_fn=generate_fn,
            rank=0,
        )
        graph_runner.replay_count = 11

        with tempfile.TemporaryDirectory() as tmp:
            status = longbench_pred._write_decode_cuda_graph_status(
                generate_fn=generate_fn,
                out_root=tmp,
                rank=0,
                before=before,
            )

        self.assertEqual(status["before"]["replay_count"], 3)
        self.assertEqual(status["replay_count"], 11)
        self.assertEqual(status["counter_delta"]["replay_count"], 8)
        self.assertEqual(status["counter_delta"]["eager_static_count"], 0)
        self.assertEqual(status["counter_delta"]["force_eager_count"], 0)

    def test_longbench_fails_if_sparseengine_graph_state_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "_sparseengine_llm"):
                longbench_pred._write_decode_cuda_graph_status(
                    generate_fn=object(),
                    out_root=tmp,
                    rank=0,
                )

if __name__ == "__main__":
    unittest.main()
