from evaluation.local_judge import parse_judge_response


def test_parse_strict_json_judge_response():
    result = parse_judge_response(
        '{"correctness":0.9,"faithfulness":1.0,'
        '"unsupported_claim":false,"critical_error":false,"reason":"ok"}'
    )
    assert result.correctness == 0.9
    assert result.faithfulness == 1.0
    assert result.unsupported_claim is False
