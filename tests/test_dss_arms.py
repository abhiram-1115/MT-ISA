import logging
import unittest

from dss_ablation import LABELS, build_arm_records, macro_f1, make_donor_map, parse_absa_output


class DSSArmTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            {"id": str(index), "text": f"review {index}", "aspect": "food", "label": LABELS[index % 3]}
            for index in range(12)
        ]
        self.explanations = {
            example["id"]: f"Explanation {example['id']}"
            for example in self.examples
        }

    def test_same_label_donors_are_deranged_and_same_label(self):
        donors = make_donor_map(self.examples, "same_label", seed=7)
        for example in self.examples:
            donor = donors[example["id"]]
            self.assertNotEqual(example["id"], donor)
            donor_example = next(item for item in self.examples if item["id"] == donor)
            self.assertEqual(example["label"], donor_example["label"])

    def test_random_donors_are_deranged(self):
        donors = make_donor_map(self.examples, "random", seed=7)
        self.assertTrue(all(receiver != donor for receiver, donor in donors.items()))

    def test_arm_sizes_and_shared_label_targets(self):
        answer_only = build_arm_records(self.examples, self.explanations, "answer_only", seed=0)
        correct = build_arm_records(self.examples, self.explanations, "correct", seed=0)
        shuffled = build_arm_records(self.examples, self.explanations, "shuffled", "random", seed=0)
        self.assertEqual(len(answer_only), len(self.examples))
        self.assertEqual(len(correct), 2 * len(self.examples))
        self.assertEqual(len(shuffled), 2 * len(self.examples))
        labels = [record["target"] for record in answer_only]
        self.assertEqual(labels, [record["target"] for record in correct if record["task"] == "label"])
        self.assertEqual(labels, [record["target"] for record in shuffled if record["task"] == "label"])

    def test_shuffled_rationales_are_a_permutation(self):
        correct = build_arm_records(self.examples, self.explanations, "correct", "same_label", seed=5)
        shuffled = build_arm_records(self.examples, self.explanations, "shuffled", "same_label", seed=5)
        correct_targets = [record["target"] for record in correct if record["task"] == "rationale"]
        shuffled_targets = [record["target"] for record in shuffled if record["task"] == "rationale"]
        self.assertCountEqual(correct_targets, shuffled_targets)

    def test_answer_only_uses_the_explanation_eligible_subset(self):
        partial = dict(list(self.explanations.items())[:5])
        answer_only = build_arm_records(self.examples, partial, "answer_only", seed=0)
        correct = build_arm_records(self.examples, partial, "correct", seed=0)
        self.assertEqual(len(answer_only), 5)
        self.assertEqual(len(correct), 10)

    def test_single_group_warns_and_falls_back_to_self(self):
        single = [{"id": "only", "text": "x", "aspect": "food", "label": "positive"}]
        with self.assertLogs("dss_ablation", level=logging.WARNING):
            donors = make_donor_map(single, "same_label", seed=0)
        self.assertEqual(donors, {"only": "only"})

    def test_nahin24_output_parser(self):
        self.assertEqual(
            parse_absa_output("[['food', 'positive'], ['service', 'neutral']]"),
            [("food", "positive"), ("service", "neutral")],
        )

    def test_manual_macro_f1_matches_hand_computation(self):
        gold = ["positive", "negative", "neutral", "neutral"]
        predicted = ["positive", "positive", "neutral", "positive"]
        self.assertAlmostEqual(macro_f1(gold, predicted), 7 / 18)


if __name__ == "__main__":
    unittest.main()
