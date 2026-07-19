"""Exponential moving-average updates for model teachers."""

from __future__ import annotations

import torch


@torch.no_grad()
def update_model_ema(teacher, student, momentum: float) -> None:
    if not 0.0 <= momentum < 1.0:
        raise ValueError("EMA momentum must be in [0, 1)")

    teacher_params = dict(teacher.named_parameters())
    student_params = dict(student.named_parameters())
    if teacher_params.keys() != student_params.keys():
        raise ValueError("EMA teacher and student parameter structures differ")

    for name, teacher_param in teacher_params.items():
        student_param = student_params[name]
        teacher_param.mul_(momentum).add_(student_param.detach(), alpha=1.0 - momentum)

    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    if teacher_buffers.keys() != student_buffers.keys():
        raise ValueError("EMA teacher and student buffer structures differ")
    for name, teacher_buffer in teacher_buffers.items():
        teacher_buffer.copy_(student_buffers[name].detach())
