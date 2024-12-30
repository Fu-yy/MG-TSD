from torch.nn.modules import loss
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from gluonts.core.component import validated

from Freq_diffusion import FrequencyDiffusion
from down_up_fourier_pool import Average_pool_upsampler, DonwSample_Fourier, moving_avg

from manifold_diffusion import SphereDiffusion
from ode_diffusion import OdeGaussianDiffusion
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
        freq_ranges: List[set],  #freq_ranges [(0,20),(0,40)]
        target_dim: int,  # target dim 1
        num_gran: int,  # the number of granularities
        conditioning_length: int,
        diff_steps: int,  # diffusion steps 100
        share_ratio_list: List[float],  # betas are shared
        freq_rate_list: List[float],  # freq_rate_list
        loss_type: str,  # L1 loss or L2 loss
        beta_end: float,  # beta_end 0.1
        beta_schedule: str,  # linear or cosine
        residual_layers: int,
        residual_channels: int,
        dilation_cycle_length: int,
        cardinality: List[int] = [1],
        embedding_dimension: int = 1,
        freq_weight_list: List[float] = [0.9, 0.1],  # freq_weight_list
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
        self.share_hidden = share_hidden
        self.weights = weights
        self.share_ratio_list = share_ratio_list
        self.num_gran = num_gran
        self.freq_ranges=freq_ranges
        self.freq_rate_list=freq_rate_list
        self.freq_weight_list=freq_weight_list
        self.split_size = [self.target_dim]*self.num_gran

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
        ) for _ in range(len(self.freq_rate_list))])  # shape: (batch_size, seq_len, num_cells)
        # self.rnn = rnn_cls(
        #     input_size=input_size,
        #     hidden_size=num_cells,
        #     num_layers=num_layers,
        #     dropout=dropout_rate,
        #     batch_first=True,
        # )   # shape: (batch_size, seq_len, num_cells)

        self.denoise_fn = EpsilonTheta(
            target_dim=target_dim,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        )  # dinosing network

        self.diffusion = GaussianDiffusion(
            self.denoise_fn,
            input_size=target_dim,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            # share ratio, new argument to control diffusion and sampling
            share_ratio_list=share_ratio_list,
            beta_schedule=beta_schedule,
        )  # diffusion network
        self.sphere_diffusion = SphereDiffusion(
            denoise_fn=self.denoise_fn,
            input_size=target_dim,
            dim=target_dim,
            num_timesteps=diff_steps,
            loss_type=loss_type,
            # share ratio, new argument to control diffusion and sampling
        )  # diffusion network

        self.ode_diffusion = OdeGaussianDiffusion(
            denoise_fn=self.denoise_fn,
            input_size=target_dim,
            loss_type=loss_type,
            # share ratio, new argument to control diffusion and sampling
        )  # diffusion network

        self.freq_diffusion = FrequencyDiffusion(
            denoise_net=self.denoise_fn,
            seq_len=48,
            dim=target_dim,
            loss_type=loss_type,
            # share ratio, new argument to control diffusion and sampling
        )  # diffusion network

        # self.diffusion = self.freq_diffusion


        self.distr_output = DiffusionOutput(
            self.diffusion, input_size=target_dim, cond_size=conditioning_length
        )  # distribution output

        self.proj_dist_args = self.distr_output.get_args_proj(
            num_cells)  # projection distribution arguments

        self.embed_dim = 1
        # self.embed = nn.Embedding(
        #     num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        # )
        self.embed = nn.ModuleList([
            nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )
            for _ in range(len(self.freq_rate_list))
        ])

        if self.scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        # kernel_list_past=[2,8]
        # self.avg_poolar_past =nn.ModuleList([
        #     Average_pool_upsampler(kernel_size=kernel)
        #     for kernel in kernel_list_past
        # ])
        # self.avg_poolar_past =nn.ModuleList([
        #     moving_avg(kernel_size=kernel,stride=1)
        #     for kernel in kernel_list_past
        # ])
        # freq_range_list_past = [(0,15),(0,40)]
        # self.fourier_mask_past = nn.ModuleList([
        #     DonwSample_Fourier(freq_range=freq_range)
        #     for freq_range in freq_range_list_past
        # ])
        #
        # kernel_list_futrue=[2,8]
        # self.avg_poolar_futrue =nn.ModuleList([
        #     Average_pool_upsampler(kernel_size=kernel)
        #     for  kernel in kernel_list_futrue
        # ])

        self.fourier_mask_futrue = nn.ModuleList([
            DonwSample_Fourier(rate=rate_item)
            for rate_item in freq_rate_list
        ])



    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:

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
            lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)

    def unroll(
        self,
        lags: torch.Tensor,
        scale: torch.Tensor,
        time_feat: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        unroll_length: int,
        index: int,
        begin_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:


        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1) # 归一化过程是将每个滞后子序列的值除以 scale，以确保它们都在相同的尺度上。lags_scaled 的形状和 lags 一样，但值是经过缩放的。 128 48 37 3

        input_lags = lags_scaled.reshape(
            (-1, unroll_length, len(self.lags_seq) * self.target_dim)
        ) #  lags_scaled 历史数据  128 48 411

        # (batch_size, target_dim, embed_dim)
        index_embeddings = self.embed[index](target_dimension_indicator) # 127 137 -- 128 137 1

        # (batch_size, seq_len, target_dim * embed_dim)
        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
            .expand(-1, unroll_length, -1, -1)
            .reshape((-1, unroll_length, self.target_dim * self.embed_dim))
        ) # 128 48 137

        # (batch_size, sub_seq_len, input_dim)
        inputs = torch.cat(
            (input_lags, repeated_index_embeddings, time_feat), dim=-1) # 将input_lags（滞后子序列）、repeated_index_embeddings（目标维度的嵌入向量）和 time_feat（时间特征）沿着最后一个维度拼接在一起，形成最终的输入张量 inputs。 #  128 48 552

        # unroll encoder
        outputs, state = self.rnn[index](inputs, begin_state)  ##  128 48 552 --> 128 48 128 -- 2 128 128
        return outputs, state, lags_scaled, inputs

    def unroll_encoder(
        self,
        past_time_feat: torch.Tensor,
        past_target_cdf,
        past_observed_values,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target_cdf,
        index,
        target_dimension_indicator: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
        sequences = sequence  # 128 216 274 --> 2*(128 216 137)  对分布划分成两部分
        # sequences = (sequence)  # 128 216 274 --> 2*(128 216 137)  对分布划分成两部分
        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )  # 128 48 137 3  list *2

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, target_dim)
        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length:, ...],
            past_observed_values[:, -self.context_length:, ...],
        )

        scales = scale # 128 1 274 --> 2*(128 1 137)
        target_dimension_indicators = target_dimension_indicator# 128 274 --> 2*(128 137)
        outputs = []
        states = []
        lags_scaled = []
        inputs = []
        # for i in range(self.num_gran):
        outputs, states, lags_scaled, inputs = self.unroll(
            lags=lags,
            scale=scales,
            time_feat=time_feat,
            # use the target_dimension_indicator 0-369
            target_dimension_indicator=target_dimension_indicators,
            unroll_length=subsequences_length,
            begin_state=None,
            index=index
        )
            # outputs.append(output)
            # states.append(state)
            # lags_scaled.append(lag_scaled)
            # inputs.append(input)

        return outputs, states, scales, lags_scaled, inputs

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
        return distr_args

    def forward(
        self,
        target_dimension_indicator: torch.Tensor, # 128 274  batch*(0,1,...,136)
        past_time_feat: torch.Tensor, # 128 192 4  时间特征
        past_target_cdf: torch.Tensor, # 128 192 274  历史数据 batchsize,history_seqlen,nvar
        past_observed_values: torch.Tensor,# 128 192 274 # 哪个位置被观察到  观察到为1  确定有效位置
        past_is_pad: torch.Tensor,# 128 192
        future_time_feat: torch.Tensor, # 128 24 4
        future_target_cdf: torch.Tensor,# 128 24 274 未来数据  batchsize,pred_len,nvar
        future_observed_values: torch.Tensor,# 128 24 274  # 哪个位置被观察到  观察到为1  确定有效位置
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
        # # 划分粒度  begin------------------------#####################################################################
        # past_observed_values_list = []
        # down_res_past_list = []
        # down_res_future_list = []
        # target_dimension_indicator_list = []
        # future_observed_values_list = []
        # for down_layer in self.fourier_mask_past:
        #     past_observed_values_list.append(past_observed_values)
        #     down_res = down_layer(torch.cat([past_target_cdf,future_target_cdf],dim=1))
        #     down_res_past  = down_res[:,0:past_target_cdf.shape[1] ,:]
        #     down_res_future  = down_res[:,past_target_cdf.shape[1]:,:]
        #     down_res_past_list.append(down_res_past)
        #     down_res_future_list.append(down_res_future)
        #     future_observed_values_list.append(future_observed_values)
        #     target_dimension_indicator_list.append(target_dimension_indicator)
        # past_target_cdf = torch.cat(down_res_past_list,dim=-1)
        # future_target_cdf = torch.cat(down_res_future_list,dim=-1)
        # past_observed_values = torch.cat(past_observed_values_list,dim=-1)
        # target_dimension_indicator = torch.cat(target_dimension_indicator_list,dim=-1)
        # future_observed_values = torch.cat(future_observed_values_list,dim=-1)
        # # 划分粒度  end------------------------#####################################################################
        #

        loss_list = []

        # for freq_range in range(len(self.freq_rate_list)):
        for index, fourer_mask_layer in enumerate(self.fourier_mask_futrue):
            # unroll the decoder in "training mode", i.e. by providing future data
            past_target_cdf_fourier = fourer_mask_layer(past_target_cdf)
            future_target_cdf_fourier = fourer_mask_layer(future_target_cdf)

            rnn_outputs, _, scale, _, _ = self.unroll_encoder(
                past_time_feat=past_time_feat,
                past_target_cdf=past_target_cdf_fourier,
                past_observed_values=past_observed_values,
                past_is_pad=past_is_pad,
                future_time_feat=future_time_feat,
                future_target_cdf=future_target_cdf_fourier,
                target_dimension_indicator=target_dimension_indicator,
                index=index
            )  # rnn_outputs -- 2* （128 48 128）  scale -- 128 1 274

            # put together target sequence
            # (batch_size, seq_len, target_dim)
            target = torch.cat(
                (past_target_cdf_fourier[:, -self.context_length:, ...],
                 future_target_cdf_fourier),
                dim=1,
            )
            target = target/scale
            targets = target

            # specifiy the beta variance for the coarse-grained dataset,
            # the scalars in the forward and backward(sampling)are different

            rnn_outputs_2 = rnn_outputs  # outputs from multiple rnns
            distr_args = self.distr_args(rnn_outputs_2)  # 128 48 100 * 2


            likelihoods = []
            likelihoods = self.diffusion.log_prob(targets, distr_args,
                                                     share_ratio=1).unsqueeze(-1)

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

            loss = weighted_average(
                likelihoods, weights=loss_weights, dim=1).mean()
            loss_list.append(loss)

        loss_sum = sum(loss_item * weight_item for loss_item,
                   weight_item in zip(loss_list, self.freq_weight_list))

        return (loss_sum, likelihoods, distr_args)


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
        index,
        begin_states: Union[List[torch.Tensor], torch.Tensor],
        # share_ratio_list: Union[List[torch.Tensor], torch.Tensor],
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

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism
        repeated_past_target_cdf_list = repeat(past_target_cdf)
        repeated_time_feat = repeat(time_feat)
        repeated_scales_list = repeat(scale)
        if self.scaling:
            self.diffusion.scale = repeated_scales_list

        repeated_target_dimension_indicator = repeat(
            target_dimension_indicator[:, :self.target_dim])

        if self.cell_type == "LSTM":
            repeated_states_list = [repeat(s, dim=1) for s in begin_states]
        else:
            repeated_states_list = repeat(begin_states, dim=1)
        # for each future time-units we draw new samples for this time-unit
        # and update the state
        future_samples_list = []
        for k in range(self.prediction_length):  # future samples from multi-gran
            # for m in range(self.num_gran):
            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target_cdf_list,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )
            rnn_outputs, repeated_states_list, _, _ = self.unroll(
                begin_state=repeated_states_list,
                lags=lags,
                scale=repeated_scales_list,
                # gran_index=m,  # use rnn which corresponding gran
                time_feat=repeated_time_feat[:, k: k + 1, ...],
                target_dimension_indicator=repeated_target_dimension_indicator,
                unroll_length=1,
                index=index
            )
            distr_args = self.distr_args(rnn_outputs)
            new_samples = self.diffusion.sample(cond=distr_args,
                                                share_ratio=1.0)
            # new_samples = self.diffusion.sample_ode_torchdiffeq(cond=distr_args,
            #                                     )

            new_samples *= repeated_scales_list

            future_samples_list.append(new_samples)
            repeated_past_target_cdf_list = torch.cat(
                (repeated_past_target_cdf_list, new_samples), dim=1)

        # (batch_size * num_samples, prediction_length, target_dim)
        samples_list = torch.cat(future_samples_list, dim=1)
        samples_reshape_list = samples_list.reshape(
            (
                -1,
                self.num_parallel_samples,
                self.prediction_length,
                self.target_dim,
            )
        )

        # samples = torch.cat(samples_reshape_list, dim=3)
        return samples_reshape_list  # output multiple forecasts

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


        # # 划分粒度  begin------------------------#####################################################################
        # # 划分粒度  begin------------------------#####################################################################
        # past_observed_values_list = []
        # down_res_past_list = []
        # down_res_future_list = []
        # target_dimension_indicator_list = []
        # for down_layer in self.fourier_mask_past:
        #     past_observed_values_list.append(past_observed_values)
        #     down_res = down_layer(past_target_cdf)
        #     down_res_past = down_res
        #     # down_res_future = down_res[:, past_target_cdf.shape[1]:, :]
        #     down_res_past_list.append(down_res_past)
        #     # down_res_future_list.append(down_res_future)
        #     target_dimension_indicator_list.append(target_dimension_indicator)
        # past_target_cdf = torch.cat(down_res_past_list, dim=-1)
        # # future_target_cdf = torch.cat(down_res_future_list, dim=-1)
        # past_observed_values = torch.cat(past_observed_values_list, dim=-1)
        # target_dimension_indicator = torch.cat(target_dimension_indicator_list, dim=-1)
        # # 划分粒度  end------------------------#####################################################################
        # # 划分粒度  end------------------------#####################################################################




        # mark padded data as unobserved
        # (batch_size, target_dim, seq_len)
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )
        forecast_value_list=[]
        for index, fourer_mask_layer in enumerate(self.fourier_mask_futrue):
            # unroll the decoder in "training mode", i.e. by providing future data
            past_target_cdf_fourier = fourer_mask_layer(past_target_cdf)
            # future_target_cdf_fourier = fourer_mask_layer(future_target_cdf)
            # unroll the decoder in "prediction mode", i.e. with past data only
            _, begin_states, scale, _, _ = self.unroll_encoder(
                past_time_feat=past_time_feat,
                past_target_cdf=past_target_cdf_fourier,
                past_observed_values=past_observed_values,
                past_is_pad=past_is_pad,
                future_time_feat=None,
                future_target_cdf=None,
                target_dimension_indicator=target_dimension_indicator,
                index=index
            )

            forecast_value = self.sampling_decoder(
                past_target_cdf=past_target_cdf_fourier,
                target_dimension_indicator=target_dimension_indicator,
                time_feat=future_time_feat,
                scale=scale,
                begin_states=begin_states,
                index=index
                # share_ratio_list=self.share_ratio_list,
            )
            forecast_value_list.append(forecast_value)
        forecast_value_sum=torch.stack(forecast_value_list,dim=-1)
        forecast_value_sum=forecast_value_sum.mean(dim=-1)
        # forecast_value_sum = sum(value_item  for value_item in zip(forecast_value_list)) / len(forecast_value_list)
        return forecast_value_sum
