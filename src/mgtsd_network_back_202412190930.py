import time

import numpy
from torch.nn.modules import loss
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from gluonts.core.component import validated

from denoise_fn_fuy import PyramidDenoisingModel
from utils import weighted_average, MeanScaler, NOPScaler
# from module import GaussianDiffusion,DiffusionOutput
from mgtsd_module import GaussianDiffusion, DiffusionOutput
from epsilon_theta import EpsilonTheta


class mgtsdTrainingNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        input_size: int,  # imput size
        num_layers: int,
        num_cells: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        dropout_rate: float,
        lags_seq: List[int],  # lag [1,24,168]
        target_dim: int,  # target dim 1
        num_gran: int,  # the number of granularities
        conditioning_length: int,
        diff_steps: int,  # diffusion steps 100
        share_ratio_list: List[float],  # betas are shared
        loss_type: str,  # L1 loss or L2 loss
        beta_end: float,  # beta_end 0.1
        beta_schedule: str,  # linear or cosine
        residual_layers: int,
        residual_channels: int,
        dilation_cycle_length: int,
        cardinality: List[int] = [1],
        embedding_dimension: int = 1,
        weights: List[float] = [0.8, 0.2],
        scaling: bool = True,
        share_hidden: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_dim = target_dim
        self.target_dim_2 = 2*target_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.history_length = history_length
        self.scaling = scaling
        self.diff_steps = diff_steps
        self.share_hidden = share_hidden
        self.weights = weights
        self.share_ratio_list = share_ratio_list
        self.num_gran = num_gran
        self.split_size = [self.target_dim]*self.num_gran
        self.sample_speed = 5
        assert len(set(lags_seq)) == len(
            lags_seq), "no duplicated lags allowed!"
        lags_seq.sort()
        self.lags_seq = lags_seq

        self.cell_type = cell_type
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]  # rnn class
        self.rnn = nn.ModuleList([rnn_cls(
            input_size=input_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        ) for _ in range(self.num_gran)])  # shape: (batch_size, seq_len, num_cells)

        # self.rnn = nn.ModuleList([rnn_cls(
        #     input_size=self.target_dim * len(self.lags_seq),
        #     hidden_size=num_cells,
        #     num_layers=num_layers,
        #     dropout=dropout_rate,
        #     batch_first=True,
        # ) for _ in range(self.num_gran)])  # shape: (batch_size, seq_len, num_cells)

        self.denoise_fn = EpsilonTheta(
            target_dim=target_dim,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        )  # dinosing network
        self.denoise_fn_pynamid = PyramidDenoisingModel(
            target_dim=target_dim,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
            embed_dim=num_cells,
        )  # dinosing network

        self.diffusion = GaussianDiffusion(
            denoise_fn=self.denoise_fn,
            # self.denoise_fn_pynamid,
            input_size=target_dim,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            # share ratio, new argument to control diffusion and sampling
            #share_ratio_list=share_ratio_list,
            beta_schedule=beta_schedule,
        )  # diffusion network

        self.distr_output = DiffusionOutput(
            self.diffusion, input_size=target_dim, cond_size=conditioning_length
        )  # distribution output

        self.proj_dist_args = self.distr_output.get_args_proj(
            num_cells)  # projection distribution arguments
        self.proj_dist_args_liunear = nn.Linear(128,100)
        self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )

        if self.scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length
            length of sequence in the T (time) dimension (axis = 1).
        indices
            list of lag indices to be used.
            eg: [1,24,168]
        subsequences_length
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices),
            containing lagged subsequences.
            Specifically, lagged[i, :, j, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: history_length + begin_index >= 0
        # that is: history_length - lag_index - sequence_length >= 0
        # hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            # shape: (batch_size, 1, sub_seq_len, C)
            lagged_values.append(
                sequence[:, begin_index:end_index, ...].unsqueeze(1))
        # shape: (batch_size, sub_seq_len, C, I) I = len(indices)=3
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)

    def unroll(
        self,
        lags: torch.Tensor,
        scale: torch.Tensor,
        time_feat: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        unroll_length: int,
        gran_index: int,
        begin_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """

        Args:
            lags (torch.Tensor): lagged sub-sequences
            scale (torch.Tensor): 归一化
            time_feat (torch.Tensor): _description_
            target_dimension_indicator (torch.Tensor): _description_
            unroll_length (int): _description_
            begin_state (Optional[Union[List[torch.Tensor], torch.Tensor]], optional): _description_. Defaults to None.

        Returns:
            Tuple[ torch.Tensor, Union[List[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor, ]: _description_
        """

        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1) # 归一化过程是将每个滞后子序列的值除以 scale，以确保它们都在相同的尺度上。lags_scaled 的形状和 lags 一样，但值是经过缩放的。 128 48 37 3

        input_lags = lags_scaled.reshape(
            (-1, unroll_length, len(self.lags_seq) * self.target_dim)
        ) #  lags_scaled 历史数据  128 48 411

        # (batch_size, target_dim, embed_dim)
        index_embeddings = self.embed(target_dimension_indicator) # 127 137 -- 128 137 1

        # (batch_size, seq_len, target_dim * embed_dim)
        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
            .expand(-1, unroll_length, -1, -1)
            .reshape((-1, unroll_length, self.target_dim * self.embed_dim))
        ) # 128 48 137

        # (batch_size, sub_seq_len, input_dim)
        inputs = torch.cat(
            (input_lags, repeated_index_embeddings, time_feat), dim=-1) # 将input_lags（滞后子序列）、repeated_index_embeddings（目标维度的嵌入向量）和 time_feat（时间特征）沿着最后一个维度拼接在一起，形成最终的输入张量 inputs。 #  128 48 552
        # inputs = input_lags
        # unroll encoder
        rnn = self.rnn[gran_index]
        outputs, state = rnn(inputs, begin_state)  ##  128 48 552 --> 128 48 128 -- 2 128 128
        return outputs, state, lags_scaled, inputs

    def unroll_encoder(
        self,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target_cdf: Optional[torch.Tensor],
        target_dimension_indicator: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Unrolls the RNN encoder over past and, if present, future data.
        Returns outputs and state of the encoder, plus the scale of
        past_target_cdf and a vector of static features that was constructed
        and fed as input to the encoder. All tensor arguments should have NTC
        layout.

        Parameters
        ----------
        past_time_feat
            Past time features (batch_size, history_length, num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        target_dimension_indicator
            Dimensionality of the time series (batch_size, target_dim)

        Returns
        -------
        outputs
            RNN outputs (batch_size, seq_len, num_cells)
        states
            RNN states. Nested list with (batch_size, num_cells) tensors with
        dimensions target_dim x num_layers x (batch_size, num_cells)
        scale
            Mean scales for the time series (batch_size, 1, target_dim)
        lags_scaled
            Scaled lags(batch_size, sub_seq_len, target_dim, num_lags)
        inputs
            inputs to the RNN

        """

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat[:, -self.context_length:, ...]
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat( # 128 48 4
                (past_time_feat[:, -self.context_length:, ...],
                 future_time_feat),
                dim=1,
            )
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1) # 128 216 274  past_target_cdf# 128 192 274, future_target_cdf# 128 24 274
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # change1: split the sequence into fine and coarse-graine dataset
        sequences = torch.split(sequence, self.split_size, dim=2)  # 128 216 274 --> 2*(128 216 137)  对分布划分成两部分
        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags = [self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        ) for sequence in sequences]  # 128 48 137 3  list *2

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, target_dim)
        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length:, ...],
            past_observed_values[:, -self.context_length:, ...],
        )

        scales = torch.split(scale, self.split_size, dim=2) # 128 1 274 --> 2*(128 1 137)
        target_dimension_indicators = torch.split(
            target_dimension_indicator, self.split_size, dim=1)# 128 274 --> 2*(128 137)
        outputs = []
        states = []
        lags_scaled = []
        inputs = []
        for i in range(self.num_gran):
            output, state, lag_scaled, input = self.unroll(
                lags=lags[i],
                scale=scales[i],
                time_feat=time_feat,
                gran_index=i,
                # use the target_dimension_indicator 0-369
                target_dimension_indicator=target_dimension_indicators[0],
                unroll_length=subsequences_length,
                begin_state=None,
            )
            outputs.append(output)
            states.append(state)
            lags_scaled.append(lag_scaled)
            inputs.append(input)

        return outputs, states, scale, lags_scaled, inputs

    def distr_args(self, rnn_outputs: torch.Tensor):
        """
        Returns the distribution of DeepVAR with respect to the RNN outputs.

        Parameters
        ----------
        rnn_outputs
            Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)

        Returns
        -------
        distr
            Distribution instance
        distr_args
            Distribution arguments
        """

        (distr_args,) = self.proj_dist_args(rnn_outputs)
        # distr_args = self.proj_dist_args_liunear(rnn_outputs)
        return distr_args

    def forward(
        self,
        target_dimension_indicator: torch.Tensor, # 128 274
        past_time_feat: torch.Tensor, # 128 192 4
        past_target_cdf: torch.Tensor, # 128 192 274
        past_observed_values: torch.Tensor,# 128 192 274
        past_is_pad: torch.Tensor,# 128 192
        future_time_feat: torch.Tensor, # 128 24 4
        future_target_cdf: torch.Tensor,# 128 24 274
        future_observed_values: torch.Tensor,# 128 24 274
    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes the loss for training DeepVAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        future_observed_values
            Indicator whether or not the future values were observed
            (batch_size, prediction_length, target_dim)

        Returns
        -------
        distr
            Loss with shape (batch_size, 1)
        likelihoods
            Likelihoods for each time step
            (batch_size, context + prediction_length, 1)
        distr_args
            Distribution arguments (context + prediction_length,
            number_of_arguments)
        """

        seq_len = self.context_length + self.prediction_length

        # unroll the decoder in "training mode", i.e. by providing future data
        rnn_outputs, _, scale, _, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
        )  # rnn_outputs -- 2* （128 48 128）  scale -- 128 1 274

        # put together target sequence
        # (batch_size, seq_len, target_dim)
        target = torch.cat(
            (past_target_cdf[:, -self.context_length:, ...],
             future_target_cdf),
            dim=1,
        )
        target = target/scale
        targets = torch.split(target, self.split_size, dim=2)

        # specifiy the beta variance for the coarse-grained dataset,
        # the scalars in the forward and backward(sampling)are different

        rnn_outputs_2 = rnn_outputs  # outputs from multiple rnns
        distr_args = [self.distr_args(rnn_output)  # 128 48 100 * 2
                      for rnn_output in rnn_outputs_2]

        likelihoods = []
        for ratio_index, share_ratio in enumerate(self.share_ratio_list):
            cur_likelihood = self.diffusion.log_prob(targets[ratio_index], distr_args[ratio_index],
                                                     share_ratio=share_ratio).unsqueeze(-1)
            likelihoods.append(cur_likelihood)

        if self.scaling:
            self.diffusion.scale = scale

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        observed_values = torch.cat(
            (
                past_observed_values[:, -self.context_length:, ...],
                future_observed_values,
            ),
            dim=1,
        )  # batch_size * seq_length * 370*2

        # mask the loss at one time step if one or more observations is missing
        # in the target dimensions (batch_size, subseq_length, 1)
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)

        loss = [weighted_average(
            likelihood, weights=loss_weights, dim=1).mean() for likelihood in likelihoods]

        loss = sum(loss_item * weight_item for loss_item,
                   weight_item in zip(loss, self.weights))
        return (loss, likelihoods, distr_args)


class mgtsdPredictionNetwork(mgtsdTrainingNetwork):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)

        print("init the prediction network")
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to
        # the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        past_target_cdf: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        begin_states: Union[List[torch.Tensor], torch.Tensor],
        share_ratio_list: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes sample paths by unrolling the RNN starting with a initial
        input and state.

        Parameters
        ----------
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        time_feat
            Dynamic features of future time series (batch_size, history_length,
            num_features)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)
        begin_states
            List of initial states for the RNN layers (batch_size, num_cells)
        Returns
        --------
        sample_paths : Tensor
            A tensor containing sampled paths. Shape: (1, num_sample_paths,
            prediction_length, target_dim).
        """

        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)
        # def repeat(tensor, dim=0):
        #     return tensor.repeat_interleave(repeats=1, dim=dim)

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism
        past_target_cdf_list = torch.split(
            past_target_cdf, self.split_size, dim=2)
        repeated_past_target_cdf_list = [
            repeat(past_target_cdf) for past_target_cdf in past_target_cdf_list]

        repeated_time_feat = repeat(time_feat)
        scales_list = torch.split(scale, self.split_size, dim=2)
        repeated_scales_list = []
        repeated_scales_list = [repeat(scale) for scale in scales_list]

        repeated_target_dimension_indicator = repeat(
            target_dimension_indicator[:, :self.target_dim])

        if self.cell_type == "LSTM":
            repeated_states_list = [repeat(s, dim=1) for s in begin_states]
        else:
            repeated_states_list = [
                repeat(begin_state, dim=1) for begin_state in begin_states]
        # for each future time-units we draw new samples for this time-unit
        # and update the state
        future_samples_list = [[] for _ in range(self.num_gran)]
        # call_count = 0
        # 记录函数开始时间
        # start_time = time.time()
        for k in range(self.prediction_length):  # future samples from multi-gran
            for m in range(self.num_gran):
                share_ratio = self.share_ratio_list[m]
                lags = self.get_lagged_subsequences(
                    sequence=repeated_past_target_cdf_list[m],
                    sequence_length=self.history_length + k,
                    indices=self.shifted_lags,
                    subsequences_length=1,
                )
                rnn_outputs, repeated_states_list[m], _, _ = self.unroll(
                    begin_state=repeated_states_list[m],
                    lags=lags,
                    scale=repeated_scales_list[m],
                    gran_index=m,  # use rnn which corresponding gran
                    time_feat=repeated_time_feat[:, k: k + 1, ...],
                    target_dimension_indicator=repeated_target_dimension_indicator,
                    unroll_length=1,
                )
                distr_args = self.distr_args(rnn_outputs)

                # -------------------------- flow begin ----------------------------

                skip = self.diff_steps // self.sample_speed
                # seq = range(0, self.diff_steps, skip)
                # nonlocal call_count
                # call_count += 1
                new_samples = self.diffusion.sample_flow_inner(cond=distr_args,
                                                    share_ratio=share_ratio,
                                                    seq_timestep=None,sample_speed=self.sample_speed)
                # new_samples = self.diffusion.sample_flow_modify(cond=distr_args,
                #                                     share_ratio=share_ratio,
                #                                     sample_speed=self.sample_speed)

                # -------------------------- flow end ----------------------------


                # new_samples = self.diffusion.sample(cond=distr_args,
                #                                     share_ratio=share_ratio)

                new_samples *= repeated_scales_list[m]

                future_samples_list[m].append(new_samples)
                repeated_past_target_cdf_list[m] = torch.cat(
                    (repeated_past_target_cdf_list[m], new_samples), dim=1)

        # (batch_size * num_samples, prediction_length, target_dim)
        samples_list = [torch.cat(future_samples, dim=1)
                        for future_samples in future_samples_list]
        samples_reshape_list = [samples.reshape((-1, self.num_parallel_samples,
                                                self.prediction_length, self.target_dim,
                                                 )) for samples in samples_list]
        # 记录函数结束时间
        # end_time = time.time()
        # 计算函数执行时间
        # execution_time = end_time - start_time
        # print(f"函数执行时间: {execution_time} 秒")
        # print(f"inner called {call_count} times.")

        samples = torch.cat(samples_reshape_list, dim=3)
        return samples  # output multiple forecasts

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts samples given the trained DeepVAR model.
        All tensors should have NTC layout.
        Parameters
        ----------
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)

        Returns
        -------
        sample_paths : Tensor
            A tensor containing sampled paths (1, num_sample_paths,
            prediction_length, target_dim).

        """

        # mark padded data as unobserved
        # (batch_size, target_dim, seq_len)
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        # unroll the decoder in "prediction mode", i.e. with past data only
        _, begin_states, scale, _, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )

        return self.sampling_decoder(
            past_target_cdf=past_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
            time_feat=future_time_feat,
            scale=scale,
            begin_states=begin_states,
            share_ratio_list=self.share_ratio_list,
        )


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='mgtsd', help='model name')
    parser.add_argument('--dataset', type=str,
                        default="solar", help='dataset name')
    # parser.add_argument('--dataset', type=str,
    #                     default="elec", help='dataset name')
    parser.add_argument('--cuda_num', type=str,
                        default='0', help='cuda number')

    parser.add_argument('--result_path', type=str,
                        default='./results/', help='result path')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-05)
    parser.add_argument('--num_cells', type=int, default=128,
                        help='number of cells in the rnn')
    parser.add_argument('--diff_steps', type=int,
                        default=100, help='diff steps')

    parser.add_argument('--input_size', type=int, default=552,
                        help='the input size of the current dataset, which is different from the original feature size but can be calculated from the original feature size.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--target_dim', type=int, default=137)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--history_length', type=int, default=192)
    parser.add_argument('--cell_type', type=str, default="GRU")
    parser.add_argument('--mg_dict', type=str, default='1',
                        help='the multi-granularity list, 1_4 means 1h and 4h, 1_4_8 means 1h, 4h and 8h')
    parser.add_argument('--num_gran', type=int, default=1,
                        help='the number of granularities, must be equal to the length of mg_dict')
    parser.add_argument('--share_ratio_list', type=str, default="1",
                        help='the share ratio list, 1_0.9, means that for the second granularity, 90% of the diffusion steps are shared with the finest granularity.')
    parser.add_argument('--weight_list', type=str, default="1.0",
                        help='the weight list, 0.9_0.1 means that the weight for the first granularity is 0.9 and the weight for the second granularity is 0.1.')
    parser.add_argument('--run_num', type=str, default="1",
                        help='the index of the run, used for the result file name')
    parser.add_argument('--wandb_space', type=str,
                        default="test", help='the space name of the wandb')
    parser.add_argument('--wandb_key', type=str, default="your wandb key",
                        help='the key of the wandb, please replace it with your own key')
    parser.add_argument('--log_metrics', type=str2bool, default="False",
                        help='whether to log the metrics to the wandb when training. it will slow down the training process')

    # 返回一个命名空间，包含传递给命令的参数
    return parser.parse_args()

if __name__ == '__main__':
    betas = numpy.linspace(0.0001, 0.02, 1000, dtype=numpy.float64)

    configs = parse_args()
    prediction_network = mgtsdPredictionNetwork(
        input_size=configs.input_size,
        target_dim=configs.target_dim,
        num_gran=configs.num_gran,
        num_layers=configs.num_layers,
        num_cells=configs.num_cells,
        cell_type=configs.cell_type,
        history_length=configs.history_length,
        context_length=24,
        prediction_length=24,
        dropout_rate=0.2,
        embedding_dimension=5,
        diff_steps=100,
        loss_type='l2',
        beta_end=0.1,
        beta_schedule='linear',
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        lags_seq=[1,24,168],
        scaling=True,
        share_ratio_list=[1],
        conditioning_length=1,
        num_parallel_samples=100,
    )
    target_dimension_indicator=torch.randn(7,137)
    past_time_feat=torch.randn(7,192,4)
    past_target_cdf=torch.randn(7,192,137)
    past_observed_values=torch.randn(7,192,137)
    past_is_pad=torch.randn(7,192)
    future_time_feat=torch.randn(7,24,4)
    res = prediction_network(target_dimension_indicator,past_time_feat,past_target_cdf,past_observed_values,past_is_pad,future_time_feat)