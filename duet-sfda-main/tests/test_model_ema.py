import unittest

import torch
import torch.nn as nn

from src.utils.model_ema import update_model_ema


class ModelEmaTest(unittest.TestCase):
    def test_updates_parameters_and_copies_buffers(self):
        teacher = nn.BatchNorm1d(2)
        student = nn.BatchNorm1d(2)
        teacher.weight.data.fill_(0.0)
        student.weight.data.fill_(2.0)
        teacher.running_mean.fill_(1.0)
        student.running_mean.fill_(3.0)

        update_model_ema(teacher, student, momentum=0.75)

        self.assertTrue(torch.allclose(teacher.weight, torch.full((2,), 0.5)))
        self.assertTrue(torch.equal(teacher.running_mean, student.running_mean))

    def test_rejects_invalid_momentum(self):
        with self.assertRaises(ValueError):
            update_model_ema(nn.Linear(2, 2), nn.Linear(2, 2), momentum=1.0)

    def test_supports_weight_norm_parameters_without_deepcopy(self):
        student = nn.utils.weight_norm(nn.Linear(3, 2))
        teacher = nn.utils.weight_norm(nn.Linear(3, 2))
        teacher.load_state_dict(student.state_dict())
        student.weight_g.data.add_(1.0)

        before = teacher.weight_g.detach().clone()
        update_model_ema(teacher, student, momentum=0.5)

        self.assertTrue(torch.all(teacher.weight_g > before))


if __name__ == "__main__":
    unittest.main()
