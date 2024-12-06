from torch.nn.modules import loss
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from gluonts.core.component import validated

from fuy_new_layer import TimeSeriesTransformerWithPatches
from utils import weighted_average, MeanScaler, NOPScaler
from mgtsd_module import GaussianDiffusion, DiffusionOutput
from epsilon_theta import EpsilonTheta


class mgtsdTrainingNetwork(nn.Module):
    @validated()
    def __init__(
            self,
            input_size: int,  # input size
            num_layers: int,
            num_cells: int,
            cell_type: str,  # This can be removed if RNN is entirely replaced
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
        self.target_dim_2 = 2 * target_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.history_length = history_length
        self.scaling = scaling
        self.share_hidden = share_hidden
        self.weights = weights
        self.share_ratio_list = share_ratio_list
        self.num_gran = num_gran
        self.split_size = [self.target_dim] * self.num_gran

        assert len(set(lags_seq)) == len(
            lags_seq), "no duplicated lags allowed!"
        lags_seq.sort()
        self.lags_seq = lags_seq

        # Remove RNN initialization
        # self.cell_type = cell_type
        # rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]  # rnn class
        # self.rnn = nn.ModuleList([rnn_cls(
        #     input_size=input_size,
        #     hidden_size=num_cells,
        #     num_layers=num_layers,
        #     dropout=dropout_rate,
        #     batch_first=True,
        # ) for _ in range(self.num_gran)])  # shape: (batch_size, seq_len, num_cells)

        # Initialize Transformer
        patch_size = 24
        seqlen = 48
        self.transformer_patches = TimeSeriesTransformerWithPatches(
            patch_size=patch_size,
            seqlen=seqlen,
            embed_dim=num_cells,
            num_heads=4,
            num_layers=num_layers,
            dim_feedforward=num_cells,
            dropout=0.1,
            max_seq_len=1000,
            input_size=input_size
        )

        self.denoise_fn = EpsilonTheta(
            target_dim=target_dim,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        )  # denoising network

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

        self.distr_output = DiffusionOutput(
            self.diffusion, input_size=target_dim, cond_size=conditioning_length
        )  # distribution output

        self.proj_dist_args = self.distr_output.get_args_proj(
            num_cells)  # projection distribution arguments
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
        """
        # As before
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
            # Remove begin_state as it's no longer needed
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Unroll using Transformer instead of RNN.
        """
        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1)  # Normalize

        input_lags = lags_scaled.reshape(
            (-1, unroll_length, len(self.lags_seq) * self.target_dim)
        )  # e.g., (batch_size, 48, 411)

        # (batch_size, target_dim, embed_dim)
        index_embeddings = self.embed(target_dimension_indicator)  # e.g., (128, 137, 1)

        # (batch_size, seq_len, target_dim * embed_dim)
        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
            .expand(-1, unroll_length, -1, -1)
            .reshape((-1, unroll_length, self.target_dim * self.embed_dim))
        )  # e.g., (128, 48, 137)

        # Concatenate input lags, index embeddings, and time features
        inputs = torch.cat(
            (input_lags, repeated_index_embeddings, time_feat), dim=-1
        )  # e.g., (128, 48, 552)

        # Pass through Transformer
        transformer_outputs = self.transformer_patches(
            sequence=input_lags,
            time_feat=time_feat,
            repeated_index_embeddings=repeated_index_embeddings
        )  # e.g., (batch_size, seq_len, num_cells)

        # Return transformer outputs instead of RNN outputs and states
        return transformer_outputs, lags_scaled, inputs

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
        List[torch.Tensor],
        torch.Tensor,
        List[torch.Tensor],
    ]:
        """
        Encode the sequences using the Transformer.
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
            time_feat = torch.cat(
                (past_time_feat[:, -self.context_length:, ...],
                 future_time_feat),
                dim=1,
            )
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # Split the sequence into granularities
        sequences = torch.split(sequence, self.split_size, dim=2)  # List of tensors
        lags = [self.get_lagged_subsequences(
            sequence=seq,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        ) for seq in sequences]  # List of lagged tensors

        # Scale based on past target CDF
        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length:, ...],
            past_observed_values[:, -self.context_length:, ...],
        )

        scales = torch.split(scale, self.split_size, dim=2)  # List of scales
        target_dimension_indicators = torch.split(
            target_dimension_indicator, self.split_size, dim=1)  # List per granularity

        outputs = []
        lags_scaled = []
        inputs = []
        for i in range(self.num_gran):
            transformer_output, lag_scaled, input_tensor = self.unroll(
                lags=lags[i],
                scale=scales[i],
                time_feat=time_feat,
                gran_index=i,
                target_dimension_indicator=target_dimension_indicators[i],
                unroll_length=subsequences_length,
                # begin_state is removed
            )
            outputs.append(transformer_output)
            lags_scaled.append(lag_scaled)
            inputs.append(input_tensor)

        return outputs, scale, lags_scaled

    def distr_args(self, transformer_outputs: torch.Tensor):
        """
        Returns the distribution arguments based on Transformer outputs.
        """
        (distr_args,) = self.proj_dist_args(transformer_outputs)
        return distr_args

    def forward(
            self,
            target_dimension_indicator: torch.Tensor,  # e.g., (128, 274)
            past_time_feat: torch.Tensor,  # e.g., (128, 192, 4)
            past_target_cdf: torch.Tensor,  # e.g., (128, 192, 274)
            past_observed_values: torch.Tensor,  # e.g., (128, 192, 274)
            past_is_pad: torch.Tensor,  # e.g., (128, 192)
            future_time_feat: torch.Tensor,  # e.g., (128, 24, 4)
            future_target_cdf: torch.Tensor,  # e.g., (128, 24, 274)
            future_observed_values: torch.Tensor,  # e.g., (128, 24, 274)
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for training.
        """
        seq_len = self.context_length + self.prediction_length

        # Encode sequences using Transformer
        rnn_outputs, scale, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
        )  # rnn_outputs are now transformer outputs

        # Concatenate past and future targets
        target = torch.cat(
            (past_target_cdf[:, -self.context_length:, ...],
             future_target_cdf),
            dim=1,
        )
        target = target / scale
        targets = torch.split(target, self.split_size, dim=2)  # Split per granularity

        # Get distribution arguments from transformer outputs
        distr_args = [self.distr_args(transformer_output)
                      for transformer_output in rnn_outputs]

        likelihoods = []
        for ratio_index, share_ratio in enumerate(self.share_ratio_list):
            cur_likelihood = self.diffusion.log_prob(
                targets[ratio_index],
                distr_args[ratio_index],
                share_ratio=share_ratio
            ).unsqueeze(-1)
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
        )  # e.g., (batch_size, seq_length, target_dim)

        # Mask the loss where observations are missing
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

        # Shifted lags for decoding
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
            self,
            past_target_cdf: torch.Tensor,
            target_dimension_indicator: torch.Tensor,
            time_feat: torch.Tensor,
            scale: torch.Tensor,
            share_ratio_list: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Sampling decoder using Transformer outputs.
        """

        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        # Split past_target_cdf per granularity
        past_target_cdf_list = torch.split(
            past_target_cdf, self.split_size, dim=2)
        repeated_past_target_cdf_list = [
            repeat(past_target_cdf) for past_target_cdf in past_target_cdf_list]

        repeated_time_feat = repeat(time_feat)
        scales_list = torch.split(scale, self.split_size, dim=2)
        repeated_scales_list = [repeat(s) for s in scales_list]

        repeated_target_dimension_indicator = repeat(
            target_dimension_indicator[:, :self.target_dim])

        # Initialize lists to store samples
        future_samples_list = [[] for _ in range(self.num_gran)]
        for k in range(self.prediction_length):  # Iterate over prediction steps
            for m in range(self.num_gran):
                share_ratio = self.share_ratio_list[m]
                # Get shifted lags for current step
                lags = self.get_lagged_subsequences(
                    sequence=repeated_past_target_cdf_list[m],
                    sequence_length=self.history_length + k,
                    indices=self.shifted_lags,
                    subsequences_length=1,
                )
                # Pass through Transformer
                transformer_output, _, _ = self.unroll(
                    lags=lags,
                    scale=repeated_scales_list[m],
                    time_feat=repeated_time_feat[:, k: k + 1, ...],
                    gran_index=m,
                    target_dimension_indicator=repeated_target_dimension_indicator,
                    unroll_length=1,
                )
                distr_args = self.distr_args(transformer_output)
                # Sample from diffusion model
                new_samples = self.diffusion.sample(
                    cond=distr_args,
                    share_ratio=share_ratio
                )

                new_samples *= repeated_scales_list[m]

                future_samples_list[m].append(new_samples)
                # Update past_target_cdf_list with new samples
                repeated_past_target_cdf_list[m] = torch.cat(
                    (repeated_past_target_cdf_list[m], new_samples), dim=1)

        # Concatenate samples from all granularities
        samples_list = [torch.cat(future_samples, dim=1)
                        for future_samples in future_samples_list]
        samples_reshape_list = [samples.reshape((
            -1, self.num_parallel_samples,
            self.prediction_length, self.target_dim,
        )) for samples in samples_list]

        # Combine samples from different granularities
        samples = torch.cat(samples_reshape_list, dim=3)
        return samples  # Shape: (batch_size, num_parallel_samples, prediction_length, target_dim)

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
        Forward pass for prediction.
        """
        # Mark padded data as unobserved
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        # Encode using Transformer without future data
        _, scale, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )

        # Sample using the decoder
        samples = self.sampling_decoder(
            past_target_cdf=past_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
            time_feat=future_time_feat,
            scale=scale,
            share_ratio_list=self.share_ratio_list,
        )

        return samples  # Shape: (batch_size, num_parallel_samples, prediction_length, target_dim)