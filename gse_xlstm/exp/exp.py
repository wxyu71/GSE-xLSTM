from typing import Any
from lightning import LightningModule
import torch

from gse_xlstm.models.base_model import BaseModel
from ..models.gse_xlstm import GSEXlstm

from ..lit.enums import Task, ForecastingTaskOptions
from torch import nn
import torchmetrics as tm

BatchType = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class ForecastingExp(LightningModule):

    def __init__(
        self,
        criterion: nn.Module,
        architecture: BaseModel | GSEXlstm,
        task: Task,
        task_options: ForecastingTaskOptions = ForecastingTaskOptions.MULTIVARIATE_2_MULTIVARIATE,
        metrics: list[tm.Metric] | dict[str, tm.Metric] = [tm.MeanSquaredError()],
    ) -> None:
        super().__init__()
        self.task_options = task_options
        self.criterion = criterion

        self.train_metrics = tm.MetricCollection(metrics, prefix="train/")
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        self.model = architecture
        self.model.task = task

        self.pred_len = self.model.pred_len
        self.seq_len = self.model.seq_len
        self.enc_in = self.model.enc_in
        self.task = task

    def _shared_step(self, batch: BatchType):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

        f_dim = (
            -1
            if self.task_options == ForecastingTaskOptions.MULTIVARIATE_2_UNIVARIATE
            else 0
        )
        outputs = outputs[:, -self.pred_len :, f_dim:].contiguous()
        batch_y = batch_y[:, -self.pred_len :, f_dim:].contiguous()
        return outputs, batch_y

    def training_step(self, batch: BatchType) -> torch.Tensor | None:
        outputs, batch_y = self._shared_step(batch)
        loss = self.criterion(outputs, batch_y)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log_dict(
            self.train_metrics(outputs, batch_y),
            prog_bar=False, on_epoch=True, on_step=False,
        )
        return loss

    def validation_step(self, batch: BatchType) -> torch.Tensor | None:
        outputs, batch_y = self._shared_step(batch)
        loss = self.criterion(outputs, batch_y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log_dict(
            self.val_metrics(outputs, batch_y),
            prog_bar=True, on_epoch=True, on_step=False,
        )
        return loss

    def test_step(self, batch: BatchType) -> None:
        outputs, batch_y = self._shared_step(batch)
        self.log_dict(
            self.test_metrics(outputs, batch_y),
            prog_bar=True, on_epoch=True, on_step=False,
        )


class LongTermForecastingExp(ForecastingExp):

    def __init__(
        self,
        criterion: nn.Module,
        architecture: BaseModel | GSEXlstm,
        task_options: ForecastingTaskOptions = ForecastingTaskOptions.MULTIVARIATE_2_MULTIVARIATE,
    ) -> None:
        super().__init__(
            criterion,
            architecture,
            Task.LONG_TERM_FORECAST,
            task_options,
            [tm.MeanAbsoluteError(), tm.MeanSquaredError()],
        )


class ShortTermForecastingExp(ForecastingExp):

    def __init__(
        self,
        criterion: nn.Module,
        architecture: BaseModel | GSEXlstm,
        task_options: ForecastingTaskOptions = ForecastingTaskOptions.MULTIVARIATE_2_MULTIVARIATE,
    ) -> None:
        super().__init__(
            criterion,
            architecture,
            Task.SHORT_TERM_FORECAST,
            task_options,
            {
                "MeanAbsoluteError": tm.MeanAbsoluteError(),
                "RootMeanSquaredError": tm.MeanSquaredError(squared=False),
                "SymmetricMeanAbsolutePercentageError": tm.SymmetricMeanAbsolutePercentageError(),
            },
        )
