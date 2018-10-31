# coding: utf-8

from typing import List

from overrides import overrides
import torch
from torch.nn import Dropout, Linear
from torch.nn import GRU

from allennlp.nn.util import last_dim_softmax, weighted_sum
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.common.params import Params


@Seq2SeqEncoder.register("stacked_gru")
class StackedGRU(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    This class implements a multiple layer GRU (RNN).
    The specificity of this implementation compared to the default one in allennlp
    (``allennlp.modules.seq2seq_encoders.Seq2SeqEncoder``) is the ability to
    specify differents hidden state size for each layer of the in the
    multiple-stacked-layers-GRU.
    Optionally, different dropouts can be individually specified for each layer of the encoder.

    Parameters
    ----------
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    hidden_sizes : ``List[int]``, required.
        The hidden state sizes of each layer of the stacked-GRU.
    num_layers : ``int``, required.
        The number of layers to stack in the encoder.
    bidirectional : ``bool``, required
        Wheter or not the layers should be bidirectional.
    dropouts : ``List[float]``, optional (default = None).
        The dropout probabilities applied to each layer. The length of this list should
        be equal to the number of layers ``num_layers``.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_sizes: List[int],
                 num_layers: int,
                 bidirectional: bool,
                 dropouts: List[float] = None) -> None:
        super(StackedGRU, self).__init__()

        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._num_layers = num_layers
        self._bidirectional = bidirectional
        self._dropouts = [0.]*num_layers if dropouts is None else dropouts

        if len(self._hidden_sizes) != self._num_layers:
            raise ValueError(f"Number of layers ({self._num_layers}) must be equal to the length of hidden state size list ({len(self._hidden_sizes)})")
        if len(self._dropouts) != self._num_layers:
            raise ValueError(f"Number of layers ({self._num_layers}) must be equal to the legnth of drouput rates list ({len(self._dropouts)})")
        
        self._output_dim = hidden_sizes[-1]
        if self._bidirectional:
            self._output_dim *= 2

        self._gru_layers: List[GRU] = [] 
        for k in range(self._num_layers):
            input_size = self._input_dim if k==0 else self._hidden_sizes[k-1]
            if self._bidirectional and (k!=0): 
                input_size *= 2

            gru_layer = GRU(input_size = input_size,
                            hidden_size = self._hidden_sizes[k],
                            dropout = self._dropouts[k],
                            num_layers = 1,
                            bidirectional = self._bidirectional)
            self.add_module(f"gru_{k}", gru_layer)				
            self._gru_layers.append(gru_layer)


    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return self._bidirectional

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        gru = self._gru_layers[0]
        outputs, _ = gru(inputs)
        
        for k in range(1, self._num_layers):
            gru = self._gru_layers[k]
            next_outputs, _ = gru(outputs)
            outputs = next_outputs
  
        return outputs

    @classmethod
    def from_params(cls, params: Params) -> 'StackedGRU':
        input_dim = params.pop_int('input_dim')
        hidden_sizes = params.pop('hidden_sizes')
        dropouts = params.pop('dropouts', None)
        num_layers = params.pop_int('num_layers')
        bidirectional = params.pop_bool('bidirectional')
        params.assert_empty(cls.__name__)

        return cls(input_dim = input_dim,
                   hidden_sizes = hidden_sizes,
                   num_layers = num_layers,
                   bidirectional = bidirectional,
                   dropouts = dropouts)