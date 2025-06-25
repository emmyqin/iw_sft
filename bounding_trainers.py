import trl
import gc
from copy import deepcopy
import torch
import torch.amp as amp
from torch import nn
from trl.models import PreTrainedModelWrapper
from trl.trainer.utils import disable_dropout_in_model
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from transformers.utils import is_peft_available, is_torch_xpu_available
import deepspeed

@dataclass
class IwSFTLoss:
    """
    Importance Weighted Self-Training Loss.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=True, num_items_in_batch=None, ref_log_probs=None, 
                 sum_over_time=True, temp=2.):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
        log_probs = - nn.functional.log_softmax(logits, dim=-1)
        
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)

        if ref_log_probs is not None:
            with torch.no_grad():      
                if sum_over_time:
                    diff = - ref_log_probs.masked_fill_(
                        padding_mask, 0.0) - nll_loss.masked_fill_(
                            padding_mask, 0.0)
                    diff = diff.sum(axis=1) / (1. - 1. * padding_mask).sum(axis=1)
                    iw = torch.exp(temp * diff)  
                    iw = torch.clamp(iw, 0., 16.)          
                else:
                    iw = torch.exp(- ref_log_probs - nll_loss.detach())
                    iw = torch.clamp(iw, 1 - 0.8, 1 + 0.8)
        else:
            iw = None

        if iw is not None:
            nll_loss *= iw

        nll_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        if num_items_in_batch:
            nll_loss = nll_loss.sum() / num_items_in_batch
        else:
            nll_loss = nll_loss.mean()
        return nll_loss
    

def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


class BoundingTrainer(trl.SFTTrainer):

    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.ref_model = self._wrap_model(ref_model, training=True)
        self.ref_model = ref_model

        if self.ref_model is not None:
            disable_dropout_in_model(self.ref_model)
            
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            self._peft_has_been_casted_to_bf16 = False
    
    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
    
    def training_step(self, model, inputs, num_items_in_batch):
        loss_step = super().training_step(model, inputs, num_items_in_batch)
        torch.cuda.empty_cache()
        gc.collect()
        # get_accelerator().empty_cache()
        return loss_step
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        print(f"------------The inputs are of shape {inputs['input_ids'].shape}-----")
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            raise ValueError("Got labels in input, expecting autoregressive training!")
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            if num_items_in_batch:
                if self.ref_model is None:
                    if not "ref_log_probs" in inputs:
                        raise ValueError("We need ref_log_probs in inputs if no ref_model is provided!")
                    ref_log_probs = inputs["ref_log_probs"]
                else:
                    with torch.no_grad():
                        ref_outputs = self.ref_model(input_ids=inputs["input_ids"], 
                                                     attention_mask=inputs["attention_mask"],
                                                     use_cache=False)
                        ref_logits = ref_outputs["logits"] if isinstance(ref_outputs, dict) else ref_outputs[0]
                        del ref_outputs
                        ref_logits = ref_logits[:, :-1, :].to(model.device)
                        ref_logits = ref_logits.detach()
                        ref_logits -= ref_logits.max(dim=-1, keepdim=True).values
                        torch.exp(ref_logits, out=ref_logits)
                        ref_logits /= ref_logits.sum(dim=-1, keepdim=True)
                        torch.log(ref_logits, out=ref_logits)
                        targets = torch.clamp(inputs['labels'][..., 1:], min=0)
                        if targets.dim() == ref_logits.dim() - 1:
                            targets = targets.unsqueeze(-1)
                        ref_log_probs = ref_logits.gather(dim=-1, index=targets)
                        del ref_logits

                loss = IwSFTLoss(epsilon=0.0)(
                    outputs, inputs['labels'], shift_labels=True, 
                    num_items_in_batch=num_items_in_batch, ref_log_probs=ref_log_probs,
                    sum_over_time=True, temp=1.0)
                
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

