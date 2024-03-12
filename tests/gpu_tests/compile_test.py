import logging
import unittest

import torch
from analyzer import Analyzer, prepare_model
from arguments import FactorArguments, ScoreArguments
from module.constants import (
    ALL_MODULE_NAME,
    COVARIANCE_FACTOR_NAMES,
    LAMBDA_FACTOR_NAMES,
)
from torch.utils import data

from tests.gpu_tests.pipeline import (
    ClassificationTask,
    construct_mnist_mlp,
    get_mnist_dataset,
)
from tests.utils import check_tensor_dict_equivalence

logging.basicConfig(level=logging.DEBUG)
OLD_FACTOR_NAME = "single_gpu"
NEW_FACTOR_NAME = "compile"
OLD_SCORE_NAME = "single_gpu"
NEW_SCORE_NAME = "compile"


class CompileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = construct_mnist_mlp()
        cls.model.load_state_dict(torch.load("model.pth"))
        cls.model = cls.model.double()

        cls.train_dataset = get_mnist_dataset(split="train", data_path="data")
        cls.train_dataset = data.Subset(cls.train_dataset, indices=list(range(200)))
        cls.eval_dataset = get_mnist_dataset(split="valid", data_path="data")
        cls.eval_dataset = data.Subset(cls.eval_dataset, indices=list(range(100)))

        cls.task = ClassificationTask()
        cls.model = prepare_model(cls.model, cls.task)
        cls.model = cls.model.cuda()
        cls.model = torch.compile(cls.model)

        cls.analyzer = Analyzer(
            analysis_name="gpu_test",
            model=cls.model,
            task=cls.task,
        )

    def test_covariance_matrices(self) -> None:
        covariance_factors = self.analyzer.load_covariance_matrices(factors_name=OLD_FACTOR_NAME)
        factor_args = FactorArguments(
            use_empirical_fisher=True,
            activation_covariance_dtype=torch.float64,
            gradient_covariance_dtype=torch.float64,
            lambda_dtype=torch.float64,
        )
        self.analyzer.fit_covariance_matrices(
            factors_name=NEW_FACTOR_NAME,
            dataset=self.train_dataset,
            factor_args=factor_args,
            per_device_batch_size=16,
            overwrite_output_dir=True,
        )
        new_covariance_factors = self.analyzer.load_covariance_matrices(factors_name=NEW_FACTOR_NAME)

        for name in COVARIANCE_FACTOR_NAMES:
            for module_name in covariance_factors[name]:
                print(f"Name: {name, module_name}")
                print(f"Previous factor: {covariance_factors[name][module_name]}")
                print(f"New factor: {new_covariance_factors[name][module_name]}")

    def test_lambda_matrices(self):
        lambda_factors = self.analyzer.load_lambda_matrices(factors_name=OLD_FACTOR_NAME)
        factor_args = FactorArguments(
            use_empirical_fisher=True,
            activation_covariance_dtype=torch.float64,
            gradient_covariance_dtype=torch.float64,
            lambda_dtype=torch.float64,
        )
        self.analyzer.fit_lambda_matrices(
            factors_name=NEW_FACTOR_NAME,
            dataset=self.train_dataset,
            factor_args=factor_args,
            per_device_batch_size=16,
            overwrite_output_dir=True,
            load_from_factors_name=OLD_FACTOR_NAME,
        )
        new_lambda_factors = self.analyzer.load_lambda_matrices(factors_name=NEW_FACTOR_NAME)

        for name in LAMBDA_FACTOR_NAMES:
            for module_name in lambda_factors[name]:
                print(f"Name: {name, module_name}")
                print(f"Previous factor: {lambda_factors[name][module_name]}")
                print(f"New factor: {new_lambda_factors[name][module_name]}")
            assert check_tensor_dict_equivalence(
                lambda_factors[name],
                new_lambda_factors[name],
                atol=1e-3,
                rtol=1e-1,
            )

    def test_pairwise_scores(self) -> None:
        pairwise_scores = self.analyzer.load_pairwise_scores(scores_name=OLD_SCORE_NAME)

        score_args = ScoreArguments(
            score_dtype=torch.float64,
            per_sample_gradient_dtype=torch.float64,
            precondition_dtype=torch.float64,
        )
        self.analyzer.compute_pairwise_scores(
            scores_name=NEW_SCORE_NAME,
            factors_name=OLD_FACTOR_NAME,
            query_dataset=self.eval_dataset,
            train_dataset=self.train_dataset,
            train_indices=list(range(42)),
            query_indices=list(range(23)),
            per_device_query_batch_size=2,
            per_device_train_batch_size=4,
            score_args=score_args,
            overwrite_output_dir=True,
        )
        new_pairwise_scores = self.analyzer.load_pairwise_scores(scores_name=NEW_SCORE_NAME)

        torch.set_printoptions(threshold=30_000)
        print(f"Previous score: {pairwise_scores[ALL_MODULE_NAME][10]}")
        print(f"Previous shape: {pairwise_scores[ALL_MODULE_NAME].shape}")
        print(f"New score: {new_pairwise_scores[ALL_MODULE_NAME][10]}")
        print(f"New shape: {new_pairwise_scores[ALL_MODULE_NAME].shape}")
        assert check_tensor_dict_equivalence(
            pairwise_scores,
            new_pairwise_scores,
            atol=1e-5,
            rtol=1e-3,
        )

    def test_self_scores(self) -> None:
        score_args = ScoreArguments(
            score_dtype=torch.float64,
            per_sample_gradient_dtype=torch.float64,
            precondition_dtype=torch.float64,
        )
        self.analyzer.compute_self_scores(
            scores_name=NEW_SCORE_NAME,
            factors_name=OLD_FACTOR_NAME,
            train_dataset=self.train_dataset,
            train_indices=list(range(42)),
            per_device_train_batch_size=4,
            score_args=score_args,
            overwrite_output_dir=True,
        )
        new_self_scores = self.analyzer.load_self_scores(scores_name=NEW_SCORE_NAME)

        self_scores = self.analyzer.load_self_scores(scores_name=OLD_SCORE_NAME)
        torch.set_printoptions(threshold=30_000)
        print(f"Previous score: {self_scores[ALL_MODULE_NAME]}")
        print(f"Previous shape: {self_scores[ALL_MODULE_NAME].shape}")
        print(f"New score: {new_self_scores[ALL_MODULE_NAME]}")
        print(f"New shape: {new_self_scores[ALL_MODULE_NAME].shape}")
        assert check_tensor_dict_equivalence(
            self_scores,
            new_self_scores,
            atol=1e-5,
            rtol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
