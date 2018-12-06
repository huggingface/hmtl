# coding: utf-8

from typing import Dict, List

import torch
from overrides import overrides

from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
import allennlp.nn.util as util


@TextFieldEmbedder.register("shortcut_connect_text_field_embedder")
class ShortcutConnectTextFieldEmbedder(TextFieldEmbedder):
    """
    This class implement a specific text field embedder that benefits from the output of 
    a ``allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder``.
    It simply concatenate two embeddings vectors: the one from the previous_encoder 
    (an ``allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder``)  and
    the one from the base_text_field_embedder 
    (an ``allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder``).
    The latter actually computes the word representation and explains the name of this class
    "ShortcutConnectTextFieldEmbedder": it will feed the input of a ``Seq2SeqEncoder`` 
    with the output of the previous_encoder and the output of the base_text_field_embedder,
    the connection with base_text_field_embedder actually circumventing the previous_encoder.
    
    Parameters
    ----------
    base_text_field_embedder : ``TextFieldEmbedder``, required
        The text field embedder that computes the word representation at the base of the model.
    previous_encoder : ``Seq2SeqEncoder``, required
        The previous seq2seqencoder.
    """

    def __init__(self, base_text_field_embedder: TextFieldEmbedder, previous_encoders: List[Seq2SeqEncoder]) -> None:
        super(ShortcutConnectTextFieldEmbedder, self).__init__()
        self._base_text_field_embedder = base_text_field_embedder
        self._previous_encoders = previous_encoders

    @overrides
    def get_output_dim(self) -> int:
        output_dim = 0
        output_dim += self._base_text_field_embedder.get_output_dim()
        output_dim += self._previous_encoders[-1].get_output_dim()

        return output_dim

    @overrides
    def forward(self, text_field_input: Dict[str, torch.Tensor], num_wrapping_dims: int = 0) -> torch.Tensor:
        text_field_embeddings = self._base_text_field_embedder.forward(text_field_input, num_wrapping_dims)
        base_representation = text_field_embeddings
        mask = util.get_text_field_mask(text_field_input)

        for encoder in self._previous_encoders:
            text_field_embeddings = encoder(text_field_embeddings, mask)
            text_field_embeddings = torch.cat([base_representation, text_field_embeddings], dim=-1)

        return torch.cat([text_field_embeddings], dim=-1)
