
import os

from torch.nn.parameter import Parameter
from openprompt.utils.logging import logger



from openprompt.data_utils import InputExample, InputFeatures
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template
from openprompt.prompts import ManualTemplate, ManualVerbalizer

import torch
from torch import nn

class SoftTemplate(Template):
    r"""This is the implementation of `The Power of Scale for Parameter-Efficient
    Prompt Tuning <https://arxiv.org/pdf/2104.08691v1.pdf>`_ . Similar to :obj:`PrefixTuningTemplate`,
    This template also does not need any textual template. Addition tokens are directly
    concatenated into the input ids. There are two initializations of the new tokens. 
    (1). random initialization. (2) initialize with the tokens of the plm (We simply take 
    the first n_tokens similar to their implementation).

    Note that this template can be simply achieved by :obj:`SoftManualTemplate`, in which
    you set `n_token` <soft> tokens template before the <text_a> will give the same result.
    """
    registered_inputflag_names = ["loss_ids", "shortenable_ids"]

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 mask_token: str = '<mask>',
                 num_tokens: int=20,
                 initialize_from_vocab: Optional[bool] = True,
                 random_range: Optional[float] = 0.5,
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer,
                         mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
        self.wte = model.get_input_embeddings()
        self.random_range = random_range
        self.num_tokens = num_tokens
        self.initialize_from_vocab = initialize_from_vocab

        self.text = text 
        self.default_text1 = "<text_a> <mask>".split()
        self.default_text2 = "<text_a> <text_b> <mask>".split()

        if self.num_tokens>0:
            self.generate_parameters()

    def on_text_set(self):
        pass

    def wrap_one_example(self, example) -> List[Dict]:  #TODO this automatic generated template may not be able to process diverse data format.
        if self.text is None:
            if example.text_b is None:
                self.text = self.default_text1
            else:
                self.text = self.default_text2
        return super().wrap_one_example(example)
    
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        r""" Filter the pretrained models's nn.Embeddding Layers. Because this layer 
        should be optimized with the pretrained models's optimizer. And should not be
        optimized twice unexpectedly. But the str(self) will still return the `wte` field,
        don't worry, it won't be optimized with the template. 
        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            if 'wte' not in elem[0]:
                yield elem



    def generate_parameters(self) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        if self.initialize_from_vocab:
            soft_embeds = self.wte.weight[:self.num_tokens].clone().detach()
        soft_embeds = torch.FloatTensor(self.num_tokens, self.wte.weight.size(1)).uniform_(-self.random_range, self.random_range)
        self.soft_embeds = nn.Parameter(soft_embeds, requires_grad=True)


    
    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        inputs_embeds = self.wte(batch['input_ids'])
        batch_size = inputs_embeds.size(0)
        if self.num_tokens>0:
            soft_embeds = self.soft_embeds.repeat(batch_size, 1, 1)
            inputs_embeds = torch.cat([soft_embeds, inputs_embeds], 1)

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        if 'attention_mask' in batch and self.num_tokens>0:
            am = batch['attention_mask']
            batch['attention_mask'] = torch.cat([torch.ones((batch_size,self.num_tokens), dtype = am.dtype,device=am.device), am], dim=-1)
        return batch
