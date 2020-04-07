import argparse

class ValidateEvaluatorArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        valid_evaluator_ids = set("metrics", "roc", "pr", "sis")

        for evaluator_id in values:
            if evaluator_id.strip() not in valid_evaluator_ids:
                raise ValueError("invalid evaluator ID {s!r}".format(s=evaluator_id.strip()))
