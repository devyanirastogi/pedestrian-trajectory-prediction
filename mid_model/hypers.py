"""
hypers.py — Trajectron++ hyperparameter dict.

This is the hyperparameters dict that MID's encoder (the imported
Trajectron / MultimodalGenerativeCVAE classes) expects. It comes from
MID/utils/trajectron_hypers.py with the encoder_dim override applied
inline (see baseline.yaml: encoder_dim=256, so each LSTM half-dim = 128).

We keep one source of truth for these so the notebook stays focused on
what's interesting — the training loop, not the boilerplate.

We strip out a few keys that are dead for our setup:
    - map_encoder block (use_map_encoding=False, so it's never read)
    - KL/CVAE annealing keys (we use the encoder for its *latent f only*;
      the CVAE prior/posterior path isn't exercised in MID's diffusion model)
However, deleting them risks AttributeError deep inside mgcvae.py — the
imported code reads many of these unconditionally. Safer to leave them in.
"""

from copy import deepcopy


def get_hyperparameters(encoder_dim: int = 256) -> dict:
    """
    Returns the hyperparams dict needed to construct the Trajectron encoder.

    Args:
        encoder_dim: the f-vector dimensionality. The four LSTM hidden dims
                     (history, future, edge, edge-influence) each get
                     encoder_dim // 2, so the concatenated latent f has
                     dimensionality encoder_dim. Default 256 matches the paper.
    """
    half = encoder_dim // 2

    hypers = {
        # ── Dataset / horizons ──
        'batch_size': 256,
        'prediction_horizon': 12,         # 4.8 s @ 0.4 s/step
        'minimum_history_length': 1,
        'maximum_history_length': 7,      # → 8 timesteps inc. current = 3.2 s

        # ── Optimization (mostly unused; we drive the loop ourselves) ──
        'grad_clip': 1.0,
        'learning_rate': 0.01,
        'learning_rate_style': 'exp',
        'learning_decay_rate': 0.9999,
        'min_learning_rate': 1e-05,

        # ── CVAE bookkeeping (referenced but inert in MID's diffusion path) ──
        'k': 1,
        'k_eval': 25,
        'kl_min': 0.07,
        'kl_weight': 100.0,
        'kl_weight_start': 0,
        'kl_decay_rate': 0.99995,
        'kl_crossover': 400,
        'kl_sigmoid_divisor': 4,
        'GMM_components': 1,
        'log_p_yt_xz_max': 6,
        'N': 1,
        'tau_init': 2.0,
        'tau_final': 0.05,
        'tau_decay_rate': 0.997,
        'use_z_logit_clipping': True,
        'z_logit_clip_start': 0.05,
        'z_logit_clip_final': 5.0,
        'z_logit_clip_crossover': 300,
        'z_logit_clip_divisor': 5,

        # ── Encoder dims (the ones that actually shape f) ──
        'enc_rnn_dim_history': half,         # 128
        'enc_rnn_dim_future':  half,
        'enc_rnn_dim_edge':    half,
        'enc_rnn_dim_edge_influence': half,
        'dec_rnn_dim': 128,
        'q_z_xy_MLP_dims': None,
        'p_z_x_MLP_dims': 32,
        'MLP_dropout_keep_prob': 0.9,
        'rnn_kwargs': {'dropout_keep_prob': 0.75},

        # ── State / pred_state schema ──
        # We feed pos+vel+acc as state, and the model predicts velocity
        # (integrated to position downstream). This matches the .pkl files
        # produced by preprocess_ethucy.
        'state': {
            'PEDESTRIAN': {
                'position':     ['x', 'y'],
                'velocity':     ['x', 'y'],
                'acceleration': ['x', 'y'],
            }
        },
        'pred_state': {'PEDESTRIAN': {'velocity': ['x', 'y']}},

        # ── Dynamics (velocity → position) ──
        'dynamic': {
            'PEDESTRIAN': {
                'name': 'SingleIntegrator',
                'distribution': False,
                'limits': {},
            }
        },

        # ── Scene graph / edges ──
        'edge_encoding': True,
        'edge_state_combine_method': 'sum',
        'edge_influence_combine_method': 'attention',
        'edge_addition_filter': [0.25, 0.5, 0.75, 1.0],
        'edge_removal_filter': [1.0, 0.0],
        'dynamic_edges': 'yes',
        'offline_scene_graph': 'yes',

        # ── Toggles ──
        'incl_robot_node': False,
        'use_map_encoding': False,
        'augment': True,
        'log_histograms': False,
        'override_attention_radius': [],
        'node_freq_mult_train': False,
        'node_freq_mult_eval': False,
        'scene_freq_mult_train': False,
        'scene_freq_mult_eval': False,
        'scene_freq_mult_viz': False,

        # ── Map encoder block (unused since use_map_encoding=False, but
        #    mgcvae.py reads the key during construction) ──
        'map_encoder': {
            'PEDESTRIAN': {
                'heading_state_index': 6,
                'patch_size': [50, 10, 50, 90],
                'map_channels': 3,
                'hidden_channels': [10, 20, 10, 1],
                'output_size': 32,
                'masks': [5, 5, 5, 5],
                'strides': [1, 1, 1, 1],
                'dropout': 0.5,
            }
        },

        # ── Misc constants the imported code reads but we don't tune ──
        'npl_rate': 0.8,
        'K': 80,
        'tao': 0.4,
    }
    return deepcopy(hypers)
