from pace.evaluation.run_effectiveness import build_parser


def test_effectiveness_parser_defaults():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--dataset",
            "hotpot",
            "--source",
            "cohort",
            "--cache-dir",
            "cache",
            "--similarity-cache",
            "similarity",
            "--output-dir",
            "outputs",
            "--calibration-manifest",
            "calibration.json",
        ]
    )

    assert args.dataset == "hotpot"
    assert args.calibration_queries == 100
    assert args.parameter_step == 0.05
    assert args.max_k == 15
    assert args.adaptive_buffer == 5
    assert args.adaptive_search_fraction == 0.9
    assert args.adaptive_min_documents == 5