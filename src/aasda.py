start_index_high = int(len(betas_high) * (1 - cur_share_ratio))
betas_sub_high = copy.deepcopy(betas_high)
betas_sub_high[:start_index_high] = 0  # share the latter part of the betas
alphas_sub_high = 1.0 - betas_sub_high
alphas_cumprod_sub_high = np.cumprod(alphas_sub_high, axis=0)
alphas_cumprod_prev_sub_high = np.append(1.0, alphas_cumprod_sub_high[:-1])
suffix = int(cur_share_ratio * 100)
self.register_buffer(
    f"alphas_cumprod_high_{suffix}", to_torch(alphas_cumprod_sub_high))
self.register_buffer(
    f"alphas_cumprod_prev_high_{suffix}", to_torch(alphas_cumprod_prev_sub_high))

self.register_buffer(f"sqrt_alphas_cumprod_high_{suffix}", to_torch(
    np.sqrt(alphas_cumprod_sub_high)))
self.register_buffer(
    f"sqrt_one_minus_alphas_cumprod_high_{suffix}", to_torch(
        np.sqrt(1.0 - alphas_cumprod_sub_high))
)
self.register_buffer(
    f"log_one_minus_alphas_cumprod_high_{suffix}", to_torch(
        np.log(1.0 - alphas_cumprod_sub_high))
)
self.register_buffer(
    f"sqrt_recip_alphas_cumprod_high_{suffix}", to_torch(
        np.sqrt(1.0 / alphas_cumprod_sub_high))
)
self.register_buffer(
    f"sqrt_recipm1_alphas_cumprod_high_{suffix}", to_torch(
        np.sqrt(1.0 / alphas_cumprod_sub_high - 1))
)
self.register_buffer(

    f"posterior_mean_coef1_{suffix}",
    to_torch(betas_sub_high * np.sqrt(alphas_cumprod_prev_sub_high) / (1.0 - alphas_cumprod_sub_high)), )

self.register_buffer(
    f"posterior_mean_coef2_high_{suffix}",

    to_torch(
        (1.0 - alphas_cumprod_prev_sub_high) *
        np.sqrt(alphas_sub_high) / (1.0 - alphas_cumprod_sub_high)

    ), )
posterior_variance_sub_high = (
        betas_sub_high * (1.0 - alphas_cumprod_prev_sub_high) / (1.0 - alphas_cumprod_sub_high))
self.register_buffer(
    f"posterior_variance_high_{suffix}", to_torch(posterior_variance_sub_high))

self.register_buffer(
    f"posterior_log_variance_clipped_high_{suffix}",
    to_torch(np.log(np.maximum(posterior_variance_sub_high, 1e-20))), )