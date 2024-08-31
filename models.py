import timm
import os
from typing import Any
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
import torch as t
from torch import nn
import transformers
import pytorch_lightning as plight
import torchmetrics
import einops as eo
from loss_functions import corn_loss, corn_label_from_logits

t.set_float32_matmul_precision("medium")
global_settings = dict(try_using_torch_compile=False)


class EnsembleModel(plight.LightningModule):
    def __init__(self, models_without_norm_df, models_with_norm_df, learning_rate=0.0002, use_simple_average=False):
        super().__init__()
        self.models_without_norm = nn.ModuleList(list(models_without_norm_df))
        self.models_with_norm = nn.ModuleList(list(models_with_norm_df))
        self.learning_rate = learning_rate
        self.use_simple_average = use_simple_average

        if not self.use_simple_average:
            self.combiner = nn.Linear(
                self.models_with_norm[0].num_classes * (len(self.models_with_norm) + len(self.models_without_norm)),
                self.models_with_norm[0].num_classes,
            )

    def forward(self, x):
        x_unnormed, x_normed = x
        if not self.use_simple_average:
            out_unnormed = t.cat([model.model_step(x_unnormed, 0)[0] for model in self.models_without_norm], dim=-1)
            out_normed = t.cat([model.model_step(x_normed, 0)[0] for model in self.models_with_norm], dim=-1)
            out_avg = self.combiner(t.cat((out_unnormed, out_normed), dim=-1))
        else:
            out_unnormed = [model.model_step(x_unnormed, 0)[0] for model in self.models_without_norm]
            out_normed = [model.model_step(x_normed, 0)[0] for model in self.models_with_norm]

            out_avg = (t.stack(out_unnormed + out_normed, dim=-1) / 2).mean(-1)
        return {"out_avg": out_avg, "out_unnormed": out_unnormed, "out_normed": out_normed}, x_unnormed[-1]

    def training_step(self, batch, batch_idx):
        out, y = self(batch)
        loss = self.models_with_norm[0]._get_loss(out["out_avg"], y, batch[0])
        self.log("train_loss", loss, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out, y = self(batch)
        preds, y_onecold, ignore_index_val = self.models_with_norm[0]._get_preds_reals(out["out_avg"], y)
        acc = torchmetrics.functional.accuracy(
            preds,
            y_onecold.to(t.long),
            ignore_index=ignore_index_val,
            num_classes=self.models_with_norm[0].num_classes,
            task="multiclass",
        )
        self.log("acc", acc * 100, prog_bar=True, sync_dist=True)
        loss = self.models_with_norm[0]._get_loss(out["out_avg"], y, batch[0])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        out, y = self(batch)
        preds, y_onecold, ignore_index_val = self.models_with_norm[0]._get_preds_reals(out["out_avg"], y)
        return preds, out, y_onecold

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr=self.learning_rate)


class TimmHeadReplace(nn.Module):
    def __init__(self, pooling=None, in_channels=512, pooling_output_dimension=1, all_identity=False) -> None:
        super().__init__()

        if all_identity:
            self.head = nn.Identity()
            self.pooling = None
        else:
            self.pooling = pooling
            if pooling is not None:
                self.pooling_output_dimension = pooling_output_dimension
                if self.pooling == "AdaptiveAvgPool2d":
                    self.pooling_layer = nn.AdaptiveAvgPool2d(pooling_output_dimension)
                elif self.pooling == "AdaptiveMaxPool2d":
                    self.pooling_layer = nn.AdaptiveMaxPool2d(pooling_output_dimension)
            self.head = nn.Flatten()

    def forward(self, x, pre_logits=False):
        if self.pooling is not None:
            if self.pooling == "stack_avg_max_attn":
                x = t.cat([layer(x) for layer in self.pooling_layer], dim=-1)
            else:
                x = self.pooling_layer(x)
        return self.head(x)


class CVModel(nn.Module):
    def __init__(
        self,
        modelname,
        in_shape,
        num_classes,
        loss_func,
        last_activation: str,
        input_padding_val=10,
        char_dims=2,
        max_seq_length=1000,
    ) -> None:
        super().__init__()
        self.modelname = modelname
        self.loss_func = loss_func
        self.in_shape = in_shape
        self.char_dims = char_dims
        self.x_shape = in_shape
        self.last_activation = last_activation
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        if self.loss_func == "OrdinalRegLoss":
            self.out_shape = 1
        else:
            self.out_shape = num_classes

        self.cv_model = timm.create_model(modelname, pretrained=True, num_classes=0)
        self.cv_model.classifier = nn.Identity()
        with t.inference_mode():
            test_out = self.cv_model(t.ones(self.in_shape, dtype=t.float32))
        self.cv_model_out_dim = test_out.shape[1]
        self.cv_model.classifier = nn.Sequential(nn.Flatten(), nn.Linear(self.cv_model_out_dim, self.max_seq_length))
        if self.out_shape == 1:
            self.logit_norm = nn.Identity()
            self.out_project = nn.Identity()
        else:
            self.logit_norm = nn.LayerNorm(self.max_seq_length)
            self.out_project = nn.Linear(1, self.out_shape)

        if last_activation == "Softmax":
            self.final_activation = nn.Softmax(dim=-1)
        elif last_activation == "Sigmoid":
            self.final_activation = nn.Sigmoid()
        elif last_activation == "LogSigmoid":
            self.final_activation = nn.LogSigmoid()
        elif last_activation == "Identity":
            self.final_activation = nn.Identity()
        else:
            raise NotImplementedError(f"{last_activation} not implemented")

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        x = self.cv_model(x)
        x = self.cv_model.classifier(x).unsqueeze(-1)
        x = self.out_project(x)
        return self.final_activation(x)


class LitModel(plight.LightningModule):
    def __init__(
        self,
        in_shape: tuple,
        hidden_dim: int,
        num_attention_heads: int,
        num_layers: int,
        loss_func: str,
        learning_rate: float,
        weight_decay: float,
        cfg: dict,
        use_lr_warmup: bool,
        use_reduce_on_plateau: bool,
        track_gradient_histogram=False,
        register_forw_hook=False,
        char_dims=2,
    ) -> None:
        super().__init__()
        if "only_use_2nd_input_stream" not in cfg:
            cfg["only_use_2nd_input_stream"] = False

        if "gamma_step_size" not in cfg:
            cfg["gamma_step_size"] = 5
        if "gamma_step_factor" not in cfg:
            cfg["gamma_step_factor"] = 0.5
        self.save_hyperparameters(
            dict(
                in_shape=in_shape,
                hidden_dim=hidden_dim,
                num_attention_heads=num_attention_heads,
                num_layers=num_layers,
                loss_func=loss_func,
                learning_rate=learning_rate,
                cfg=cfg,
                x_shape=in_shape,
                num_classes=cfg["num_classes"],
                use_lr_warmup=use_lr_warmup,
                num_warmup_steps=cfg["num_warmup_steps"],
                use_reduce_on_plateau=use_reduce_on_plateau,
                weight_decay=weight_decay,
                track_gradient_histogram=track_gradient_histogram,
                register_forw_hook=register_forw_hook,
                char_dims=char_dims,
                remove_timm_classifier_head_pooling=cfg["remove_timm_classifier_head_pooling"],
                change_pooling_for_timm_head_to=cfg["change_pooling_for_timm_head_to"],
                chars_conv_pooling_out_dim=cfg["chars_conv_pooling_out_dim"],
            )
        )
        self.model_to_use = cfg["model_to_use"]
        self.num_classes = cfg["num_classes"]
        self.x_shape = in_shape
        self.in_shape = in_shape
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers

        self.use_lr_warmup = use_lr_warmup
        self.num_warmup_steps = cfg["num_warmup_steps"]
        self.warmup_exponent = cfg["warmup_exponent"]

        self.use_reduce_on_plateau = use_reduce_on_plateau
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.using_one_hot_targets = cfg["one_hot_y"]
        self.track_gradient_histogram = track_gradient_histogram
        self.register_forw_hook = register_forw_hook
        if self.loss_func == "OrdinalRegLoss":
            self.ord_reg_loss_max = cfg["ord_reg_loss_max"]
            self.ord_reg_loss_min = cfg["ord_reg_loss_min"]

        self.num_lin_layers = cfg["num_lin_layers"]
        self.linear_activation = cfg["linear_activation"]
        self.last_activation = cfg["last_activation"]

        self.max_seq_length = cfg["manual_max_sequence_for_model"]

        self.use_char_embed_info = cfg["use_embedded_char_pos_info"]

        self.method_chars_into_model = cfg["method_chars_into_model"]
        self.source_for_pretrained_cv_model = cfg["source_for_pretrained_cv_model"]
        self.method_to_include_char_positions = cfg["method_to_include_char_positions"]

        self.char_dims = char_dims
        self.char_sequence_length = cfg["max_len_chars_list"] if self.use_char_embed_info else 0

        self.chars_conv_lr_reduction_factor = cfg["chars_conv_lr_reduction_factor"]
        if self.use_char_embed_info:
            self.chars_bert_reduction_factor = cfg["chars_bert_reduction_factor"]

        self.use_in_projection_bias = cfg["use_in_projection_bias"]
        self.add_layer_norm_to_in_projection = cfg["add_layer_norm_to_in_projection"]

        self.hidden_dropout_prob = cfg["hidden_dropout_prob"]
        self.layer_norm_after_in_projection = cfg["layer_norm_after_in_projection"]
        self.method_chars_into_model = cfg["method_chars_into_model"]
        self.input_padding_val = cfg["input_padding_val"]
        self.cv_char_modelname = cfg["cv_char_modelname"]
        self.char_plot_shape = cfg["char_plot_shape"]

        self.remove_timm_classifier_head_pooling = cfg["remove_timm_classifier_head_pooling"]
        self.change_pooling_for_timm_head_to = cfg["change_pooling_for_timm_head_to"]
        self.chars_conv_pooling_out_dim = cfg["chars_conv_pooling_out_dim"]

        self.add_layer_norm_to_char_mlp = cfg["add_layer_norm_to_char_mlp"]
        if "profile_torch_run" in cfg:
            self.profile_torch_run = cfg["profile_torch_run"]
        else:
            self.profile_torch_run = False
        if self.loss_func == "OrdinalRegLoss":
            self.out_shape = 1
        else:
            self.out_shape = cfg["num_classes"]

        if not self.hparams.cfg["only_use_2nd_input_stream"]:
            if (
                self.method_chars_into_model == "dense"
                and self.use_char_embed_info
                and self.method_to_include_char_positions == "concat"
            ):
                self.project = nn.Linear(self.x_shape[-1], self.hidden_dim // 2, bias=self.use_in_projection_bias)
            elif (
                self.method_chars_into_model == "bert"
                and self.use_char_embed_info
                and self.method_to_include_char_positions == "concat"
            ):
                self.hidden_dim_chars = self.hidden_dim // 2
                self.project = nn.Linear(self.x_shape[-1], self.hidden_dim_chars, bias=self.use_in_projection_bias)
            elif (
                self.method_chars_into_model == "resnet"
                and self.method_to_include_char_positions == "concat"
                and self.use_char_embed_info
            ):
                self.project = nn.Linear(self.x_shape[-1], self.hidden_dim // 2, bias=self.use_in_projection_bias)
            elif self.model_to_use == "cv_only_model":
                self.project = nn.Identity()
            else:
                self.project = nn.Linear(self.x_shape[-1], self.hidden_dim, bias=self.use_in_projection_bias)
            if self.add_layer_norm_to_in_projection:
                self.project = nn.Sequential(
                    nn.Linear(self.project.in_features, self.project.out_features, bias=self.use_in_projection_bias),
                    nn.LayerNorm(self.project.out_features),
                )

        if hasattr(self, "project") and "posix" in os.name and global_settings["try_using_torch_compile"]:
            self.project = t.compile(self.project)

        if self.use_char_embed_info:
            self._create_char_model()

        if self.layer_norm_after_in_projection:
            if self.hparams.cfg["only_use_2nd_input_stream"]:
                self.layer_norm_in = nn.LayerNorm(self.hidden_dim // 2)
            else:
                self.layer_norm_in = nn.LayerNorm(self.hidden_dim)

            if "posix" in os.name and global_settings["try_using_torch_compile"]:
                self.layer_norm_in = t.compile(self.layer_norm_in)

        self._create_main_seq_model(cfg)

        if register_forw_hook:
            self.register_hooks()
        if self.hparams.cfg["only_use_2nd_input_stream"]:
            linear_in_dim = self.hidden_dim // 2
        else:
            linear_in_dim = self.hidden_dim

        if self.num_lin_layers == 1:
            self.linear = nn.Linear(linear_in_dim, self.out_shape)
        else:
            lin_layers = []
            for _ in range(self.num_lin_layers - 1):
                lin_layers.extend(
                    [
                        nn.Linear(linear_in_dim, linear_in_dim),
                        getattr(nn, self.linear_activation)(),
                    ]
                )
            self.linear = nn.Sequential(*lin_layers, nn.Linear(linear_in_dim, self.out_shape))

        if "posix" in os.name and global_settings["try_using_torch_compile"]:
            self.linear = t.compile(self.linear)

        if self.last_activation == "Softmax":
            self.final_activation = nn.Softmax(dim=-1)
        elif self.last_activation == "Sigmoid":
            self.final_activation = nn.Sigmoid()
        elif self.last_activation == "Identity":
            self.final_activation = nn.Identity()
        else:
            raise NotImplementedError(f"{self.last_activation} not implemented")

        if self.profile_torch_run:
            self.profilerr = t.profiler.profile(
                schedule=t.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
                on_trace_ready=t.profiler.tensorboard_trace_handler("tblogs"),
                with_stack=True,
                record_shapes=True,
                profile_memory=False,
            )

    def _create_main_seq_model(self, cfg):
        if self.hparams.cfg["only_use_2nd_input_stream"]:
            hidden_dim = self.hidden_dim // 2
        else:
            hidden_dim = self.hidden_dim
        if self.model_to_use == "BERT":
            self.bert_config = transformers.BertConfig(
                vocab_size=self.x_shape[-1],
                hidden_size=hidden_dim,
                num_hidden_layers=self.num_layers,
                intermediate_size=hidden_dim,
                num_attention_heads=self.num_attention_heads,
                max_position_embeddings=self.max_seq_length,
            )
            self.bert_model = transformers.BertModel(self.bert_config)
        elif self.model_to_use == "cv_only_model":
            self.bert_model = CVModel(
                modelname=cfg["cv_modelname"],
                in_shape=self.in_shape,
                num_classes=cfg["num_classes"],
                loss_func=cfg["loss_function"],
                last_activation=cfg["last_activation"],
                input_padding_val=cfg["input_padding_val"],
                char_dims=self.char_dims,
                max_seq_length=cfg["manual_max_sequence_for_model"],
            )
        else:
            raise NotImplementedError(f"{self.model_to_use} not implemented")
        if "posix" in os.name and global_settings["try_using_torch_compile"]:
            self.bert_model = t.compile(self.bert_model)
        return 0

    def _create_char_model(self):
        if self.method_chars_into_model == "dense":
            self.chars_project_0 = nn.Linear(self.char_dims, 1, bias=self.use_in_projection_bias)
            if "posix" in os.name and global_settings["try_using_torch_compile"]:
                self.chars_project_0 = t.compile(self.chars_project_0)
            if self.method_to_include_char_positions == "concat":
                self.chars_project_1 = nn.Linear(
                    self.char_sequence_length, self.hidden_dim // 2, bias=self.use_in_projection_bias
                )
            else:
                self.chars_project_1 = nn.Linear(
                    self.char_sequence_length, self.hidden_dim, bias=self.use_in_projection_bias
                )

            if "posix" in os.name and global_settings["try_using_torch_compile"]:
                self.chars_project_1 = t.compile(self.chars_project_1)
        elif not self.method_chars_into_model == "resnet":
            self.chars_project = nn.Linear(self.char_dims, self.hidden_dim_chars, bias=self.use_in_projection_bias)
            if "posix" in os.name and global_settings["try_using_torch_compile"]:
                self.chars_project = t.compile(self.chars_project)

        if self.method_chars_into_model == "bert":
            if not hasattr(self, "hidden_dim_chars"):
                if self.hidden_dim // self.chars_bert_reduction_factor > 1:
                    self.hidden_dim_chars = self.hidden_dim // self.chars_bert_reduction_factor
                else:
                    self.hidden_dim_chars = self.hidden_dim
            self.num_attention_heads_chars = self.hidden_dim_chars // (self.hidden_dim // self.num_attention_heads)
            self.chars_bert_config = transformers.BertConfig(
                vocab_size=self.x_shape[-1],
                hidden_size=self.hidden_dim_chars,
                num_hidden_layers=self.num_layers,
                intermediate_size=self.hidden_dim_chars,
                num_attention_heads=self.num_attention_heads_chars,
                max_position_embeddings=self.char_sequence_length + 1,
                num_labels=1,
            )
            self.chars_bert = transformers.BertForSequenceClassification(self.chars_bert_config)

            if "posix" in os.name and global_settings["try_using_torch_compile"]:
                self.chars_bert = t.compile(self.chars_bert)
            self.chars_project_class_output = nn.Linear(1, self.hidden_dim_chars, bias=self.use_in_projection_bias)
            if "posix" in os.name and global_settings["try_using_torch_compile"]:
                self.chars_project_class_output = t.compile(self.chars_project_class_output)
        elif self.method_chars_into_model == "resnet":
            if self.source_for_pretrained_cv_model == "timm":
                self.chars_conv = timm.create_model(
                    self.cv_char_modelname,
                    pretrained=True,
                    num_classes=0,  # remove classifier nn.Linear
                )
                if self.remove_timm_classifier_head_pooling:
                    self.chars_conv.head = TimmHeadReplace(all_identity=True)
                    with t.inference_mode():
                        test_out = self.chars_conv(
                            t.ones((1, 3, self.char_plot_shape[0], self.char_plot_shape[1]), dtype=t.float32)
                        )
                    if test_out.ndim > 3:
                        self.chars_conv.head = TimmHeadReplace(
                            self.change_pooling_for_timm_head_to,
                            test_out.shape[1],
                        )
            elif self.source_for_pretrained_cv_model == "huggingface":
                self.chars_conv = transformers.AutoModelForImageClassification.from_pretrained(self.cv_char_modelname)
            elif self.source_for_pretrained_cv_model == "torch_hub":
                self.chars_conv = t.hub.load(*self.cv_char_modelname.split(","))

            if hasattr(self.chars_conv, "classifier"):
                self.chars_conv.classifier = nn.Identity()
            elif hasattr(self.chars_conv, "cls_classifier"):
                self.chars_conv.cls_classifier = nn.Identity()
            elif hasattr(self.chars_conv, "fc"):
                self.chars_conv.fc = nn.Identity()

            if hasattr(self.chars_conv, "distillation_classifier"):
                self.chars_conv.distillation_classifier = nn.Identity()
            with t.inference_mode():
                test_out = self.chars_conv(
                    t.ones((1, 3, self.char_plot_shape[0], self.char_plot_shape[1]), dtype=t.float32)
                )
            if hasattr(test_out, "last_hidden_state"):
                self.chars_conv_out_dim = test_out.last_hidden_state.shape[1]
            elif hasattr(test_out, "logits"):
                self.chars_conv_out_dim = test_out.logits.shape[1]
            elif isinstance(test_out, list):
                self.chars_conv_out_dim = test_out[0].shape[1]
            else:
                self.chars_conv_out_dim = test_out.shape[1]

            char_lin_layers = [nn.Flatten(), nn.Linear(self.chars_conv_out_dim, self.hidden_dim // 2)]
            if self.add_layer_norm_to_char_mlp:
                char_lin_layers.append(nn.LayerNorm(self.hidden_dim // 2))
            self.chars_classifier = nn.Sequential(*char_lin_layers)
            if hasattr(self.chars_conv, "distillation_classifier"):
                self.chars_conv.distillation_classifier = nn.Sequential(
                    nn.Flatten(), nn.Linear(self.chars_conv_out_dim, self.hidden_dim // 2)
                )

            if "posix" in os.name and global_settings["try_using_torch_compile"]:
                self.chars_classifier = t.compile(self.chars_classifier)
            if "posix" in os.name and global_settings["try_using_torch_compile"]:
                self.chars_conv = t.compile(self.chars_conv)
        return 0

    def register_hooks(self):
        def add_to_tb(layer):
            def hook(model, input, output):
                if hasattr(output, "detach"):
                    for logger in self.loggers:
                        if hasattr(logger.experiment, "add_histogram"):
                            logger.experiment.add_histogram(
                                tag=f"{layer}_{str(list(output.shape))}",
                                values=output.detach(),
                                global_step=self.trainer.global_step,
                            )

            return hook

        for layer_id, layer in dict([*self.named_modules()]).items():
            layer.register_forward_hook(add_to_tb(f"act_{layer_id}"))

    def on_after_backward(self) -> None:
        if self.track_gradient_histogram:
            if self.trainer.global_step % 200 == 0:
                for logger in self.loggers:
                    if hasattr(logger.experiment, "add_histogram"):
                        for layer_id, layer in dict([*self.named_modules()]).items():
                            parameters = layer.parameters()
                            for idx2, p in enumerate(parameters):
                                grad_val = p.grad
                                if grad_val is not None:
                                    grad_name = f"grad_{idx2}_{layer_id}_{str(list(p.grad.shape))}"
                                    logger.experiment.add_histogram(
                                        tag=grad_name, values=grad_val, global_step=self.trainer.global_step
                                    )

        return super().on_after_backward()

    def _fold_in_seq_dim(self, out, y):
        batch_size, seq_len, num_classes = out.shape
        out = eo.rearrange(out, "b s c -> (b s) c", s=seq_len)
        if y is None:
            return out, None
        if len(y.shape) > 2:
            y = eo.rearrange(y, "b s c -> (b s) c", s=seq_len)
        else:
            y = eo.rearrange(y, "b s -> (b s)", s=seq_len)
        return out, y

    def _get_loss(self, out, y, batch):
        attention_mask = batch[-2]
        if self.loss_func == "BCELoss":
            if self.last_activation == "Identity":
                loss = t.nn.functional.binary_cross_entropy_with_logits(out, y, reduction="none")
            else:
                loss = t.nn.functional.binary_cross_entropy(out, y, reduction="none")

            replace_tensor = t.zeros(loss[1, 1, :].shape, device=loss.device, dtype=loss.dtype, requires_grad=False)
            loss[~attention_mask.bool()] = replace_tensor
            loss = loss.mean()
        elif self.loss_func == "CrossEntropyLoss":
            if len(out.shape) > 2:
                out, y = self._fold_in_seq_dim(out, y)
                loss = t.nn.functional.cross_entropy(out, y, reduction="mean", ignore_index=-100)
            else:
                loss = t.nn.functional.cross_entropy(out, y, reduction="mean", ignore_index=-100)

        elif self.loss_func == "OrdinalRegLoss":
            loss = t.nn.functional.mse_loss(out, y, reduction="none")
            loss = loss[attention_mask.bool()].sum() * 10.0 / attention_mask.sum()
        elif self.loss_func == "corn_loss":
            out, y = self._fold_in_seq_dim(out, y)
            loss = corn_loss(out, y.squeeze(), self.out_shape)
        else:
            raise ValueError("Loss Function not reckognized")
        return loss

    def training_step(self, batch, batch_idx):
        if self.profile_torch_run:
            self.profilerr.step()
        out, y = self.model_step(batch, batch_idx)
        loss = self._get_loss(out, y, batch)
        self.log("train_loss", loss, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def forward(*args):
        return forward(args[0], args[1:])

    def model_step(self, batch, batch_idx):
        out = self.forward(batch)
        return out, batch[-1]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        optimizer.step(closure=optimizer_closure)

        if self.use_lr_warmup and self.hparams["cfg"]["lr_scheduling"] != "OneCycleLR":
            if self.trainer.global_step < self.num_warmup_steps:
                lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.num_warmup_steps) ** self.warmup_exponent
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.hparams.learning_rate
        if self.trainer.global_step % 10 == 0 or self.trainer.global_step == 0:
            for idx, pg in enumerate(optimizer.param_groups):
                self.log(f"lr_{idx}", pg["lr"], prog_bar=True, sync_dist=True)

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any | None) -> None:
        if self.use_lr_warmup and self.hparams["cfg"]["lr_scheduling"] != "OneCycleLR":
            if self.trainer.global_step > self.num_warmup_steps:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def _get_preds_reals(self, out, y):
        if self.loss_func == "corn_loss":
            seq_len = out.shape[1]
            out, y = self._fold_in_seq_dim(out, y)
            preds = corn_label_from_logits(out)
            preds = eo.rearrange(preds, "(b s) -> b s", s=seq_len)
            if y is not None:
                y = eo.rearrange(y.squeeze(), "(b s) -> b s", s=seq_len)

        elif self.loss_func == "OrdinalRegLoss":
            preds = out * (self.ord_reg_loss_max - self.ord_reg_loss_min)
            preds = (preds + self.ord_reg_loss_min).round().to(t.long)

        else:
            preds = t.argmax(out, dim=-1)
        if y is None:
            return preds, y, -100
        else:
            if self.using_one_hot_targets:
                y_onecold = t.argmax(y, dim=-1)
                ignore_index_val = 0
            elif self.loss_func == "OrdinalRegLoss":
                y_onecold = (y * self.num_classes).round().to(t.long)

                y_onecold = y * (self.ord_reg_loss_max - self.ord_reg_loss_min)
                y_onecold = (y_onecold + self.ord_reg_loss_min).round().to(t.long)
                ignore_index_val = t.min(y_onecold).to(t.long)
            else:
                y_onecold = y
                ignore_index_val = -100

            if len(preds.shape) > len(y_onecold.shape):
                preds = preds.squeeze()
            return preds, y_onecold, ignore_index_val

    def validation_step(self, batch, batch_idx):
        out, y = self.model_step(batch, batch_idx)
        preds, y_onecold, ignore_index_val = self._get_preds_reals(out, y)

        if self.loss_func == "OrdinalRegLoss":
            y_onecold = y_onecold.flatten()
            preds = preds.flatten()[y_onecold != ignore_index_val]
            y_onecold = y_onecold[y_onecold != ignore_index_val]
            acc = (preds == y_onecold).sum() / len(y_onecold)
        else:
            acc = torchmetrics.functional.accuracy(
                preds,
                y_onecold.to(t.long),
                ignore_index=ignore_index_val,
                num_classes=self.num_classes,
                task="multiclass",
            )
            self.log("acc", acc * 100, prog_bar=True, sync_dist=True)
        loss = self._get_loss(out, y, batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def predict_step(self, batch, batch_idx):
        out, y = self.model_step(batch, batch_idx)
        preds, y_onecold, ignore_index_val = self._get_preds_reals(out, y)
        return preds, y_onecold

    def configure_optimizers(self):
        params = list(self.named_parameters())

        def is_chars_conv(n):
            if "chars_conv" not in n:
                return False
            if "chars_conv" in n and "classifier" in n:
                return False
            else:
                return True

        grouped_parameters = [
            {
                "params": [p for n, p in params if is_chars_conv(n)],
                "lr": self.learning_rate / self.chars_conv_lr_reduction_factor,
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in params if not is_chars_conv(n)],
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
            },
        ]
        opti = t.optim.AdamW(grouped_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.use_reduce_on_plateau:
            opti_dict = {
                "optimizer": opti,
                "lr_scheduler": {
                    "scheduler": t.optim.lr_scheduler.ReduceLROnPlateau(opti, mode="min", patience=2, factor=0.5),
                    "monitor": "val_loss",
                    "frequency": 1,
                    "interval": "epoch",
                },
            }
            return opti_dict
        else:
            cfg = self.hparams["cfg"]
            if cfg["use_reduce_on_plateau"]:
                scheduler = None
            elif cfg["lr_scheduling"] == "multistep":
                scheduler = t.optim.lr_scheduler.MultiStepLR(
                    opti, milestones=cfg["multistep_milestones"], gamma=cfg["gamma_multistep"], verbose=False
                )
                interval = "step" if cfg["use_training_steps_for_end_and_lr_decay"] else "epoch"
            elif cfg["lr_scheduling"] == "StepLR":
                scheduler = t.optim.lr_scheduler.StepLR(
                    opti, step_size=cfg["gamma_step_size"], gamma=cfg["gamma_step_factor"]
                )
                interval = "step" if cfg["use_training_steps_for_end_and_lr_decay"] else "epoch"
            elif cfg["lr_scheduling"] == "anneal":
                scheduler = t.optim.lr_scheduler.CosineAnnealingLR(
                    opti, 250, eta_min=cfg["min_lr_anneal"], last_epoch=-1, verbose=False
                )
                interval = "step"
            elif cfg["lr_scheduling"] == "ExponentialLR":
                scheduler = t.optim.lr_scheduler.ExponentialLR(opti, gamma=cfg["lr_sched_exp_fac"])
                interval = "step"
            else:
                scheduler = None
            if scheduler is None:
                return [opti]
            else:
                opti_dict = {
                    "optimizer": opti,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "global_step",
                        "frequency": 1,
                        "interval": interval,
                    },
                }
                return opti_dict

    def on_fit_start(self) -> None:
        if self.profile_torch_run:
            self.profilerr.start()
        return super().on_fit_start()

    def on_fit_end(self) -> None:
        if self.profile_torch_run:
            self.profilerr.stop()
        return super().on_fit_end()


def prep_model_input(self, batch):
    if len(batch) == 1:
        batch = batch[0]
    if self.use_char_embed_info:
        if len(batch) == 5:
            x, chars_coords, ims, attention_mask, _ = batch
        elif batch[1].ndim == 4:
            x, ims, attention_mask, _ = batch
        else:
            x, chars_coords, attention_mask, _ = batch
        padding_list = None
    else:
        if len(batch) > 3:
            x = batch[0]
            y = batch[-1]
            attention_mask = batch[1]
        else:
            x, attention_mask, y = batch

    if self.model_to_use != "cv_only_model" and not self.hparams.cfg["only_use_2nd_input_stream"]:
        x_embedded = self.project(x)
    else:
        x_embedded = x
    if self.use_char_embed_info:
        if self.method_chars_into_model == "dense":
            bool_mask = chars_coords == self.input_padding_val
            bool_mask = bool_mask[:, :, 0]
            chars_coords_projected = self.chars_project_0(chars_coords).squeeze(-1)
            chars_coords_projected = chars_coords_projected * bool_mask
            if self.chars_project_1.in_features == chars_coords_projected.shape[-1]:
                chars_coords_projected = self.chars_project_1(chars_coords_projected)
            else:
                chars_coords_projected = chars_coords_projected.mean(dim=-1)
                chars_coords_projected = chars_coords_projected.unsqueeze(1).repeat(1, x_embedded.shape[2])
        elif self.method_chars_into_model == "bert":
            chars_mask = chars_coords != self.input_padding_val
            chars_mask = t.cat(
                (
                    t.ones(chars_mask[:, :1, 0].shape, dtype=t.long, device=chars_coords.device),
                    chars_mask[:, :, 0].to(t.long),
                ),
                dim=1,
            )
            chars_coords_projected = self.chars_project(chars_coords)

            position_ids = t.arange(
                0, chars_coords_projected.shape[1] + 1, dtype=t.long, device=chars_coords_projected.device
            )
            token_type_ids = t.zeros(
                (chars_coords_projected.size()[0], chars_coords_projected.size()[1] + 1),
                dtype=t.long,
                device=chars_coords_projected.device,
            )  # +1 for CLS
            chars_coords_projected = t.cat(
                (t.ones_like(chars_coords_projected[:, :1, :]), chars_coords_projected), dim=1
            )  # to add CLS token
            chars_coords_projected = self.chars_bert(
                position_ids=position_ids,
                inputs_embeds=chars_coords_projected,
                token_type_ids=token_type_ids,
                attention_mask=chars_mask,
            )
            if hasattr(chars_coords_projected, "last_hidden_state"):
                chars_coords_projected = chars_coords_projected.last_hidden_state[:, 0, :]
            elif hasattr(chars_coords_projected, "logits"):
                chars_coords_projected = chars_coords_projected.logits
            else:
                chars_coords_projected = chars_coords_projected.hidden_states[-1][:, 0, :]
        elif self.method_chars_into_model == "resnet":
            chars_conv_out = self.chars_conv(ims)
            if isinstance(chars_conv_out, list):
                chars_conv_out = chars_conv_out[0]
            if hasattr(chars_conv_out, "logits"):
                chars_conv_out = chars_conv_out.logits
            chars_coords_projected = self.chars_classifier(chars_conv_out)

        chars_coords_projected = chars_coords_projected.unsqueeze(1).repeat(1, x_embedded.shape[1], 1)
        if hasattr(self, "chars_project_class_output"):
            chars_coords_projected = self.chars_project_class_output(chars_coords_projected)

        if self.hparams.cfg["only_use_2nd_input_stream"]:
            x_embedded = chars_coords_projected
        elif self.method_to_include_char_positions == "concat":
            x_embedded = t.cat((x_embedded, chars_coords_projected), dim=-1)
        else:
            x_embedded = x_embedded + chars_coords_projected
    return x_embedded, attention_mask


def forward(self, batch):
    prepped_input = prep_model_input(self, batch)

    if len(batch) > 5:
        x_embedded, padding_list, attention_mask, attention_mask_for_prediction = prepped_input
    elif len(batch) > 2:
        x_embedded, attention_mask = prepped_input
    else:
        x_embedded = prepped_input[0]
        attention_mask = prepped_input[-1]

    position_ids = t.arange(0, x_embedded.shape[1], dtype=t.long, device=x_embedded.device)
    token_type_ids = t.zeros(x_embedded.size()[:-1], dtype=t.long, device=x_embedded.device)

    if self.layer_norm_after_in_projection:
        x_embedded = self.layer_norm_in(x_embedded)

    if self.model_to_use == "LSTM":
        bert_out = self.bert_model(x_embedded)
    elif self.model_to_use in ["ProphetNet", "T5", "FunnelModel"]:
        bert_out = self.bert_model(inputs_embeds=x_embedded, attention_mask=attention_mask)
    elif self.model_to_use == "xBERT":
        bert_out = self.bert_model(x_embedded, mask=attention_mask.to(bool))
    elif self.model_to_use == "cv_only_model":
        bert_out = self.bert_model(x_embedded)
    else:
        bert_out = self.bert_model(
            position_ids=position_ids,
            inputs_embeds=x_embedded,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
    if hasattr(bert_out, "last_hidden_state"):
        last_hidden_state = bert_out.last_hidden_state
        out = self.linear(last_hidden_state)
    elif hasattr(bert_out, "logits"):
        out = bert_out.logits
    else:
        out = bert_out
    out = self.final_activation(out)
    return out
