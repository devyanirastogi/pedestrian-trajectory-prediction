# Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion

A reimplementation of the MID framework ([Gu et al., 2022](https://arxiv.org/abs/2203.13777)) for pedestrian trajectory prediction on the ETH/UCY benchmark.

## 1. Problem Statement

Predicting where pedestrians will walk in the next few seconds is important for applications like self-driving cars and social robots. The challenge is that human motion is inherently uncertain — given the same observed path, a person could turn left, keep going straight, or stop entirely. So the prediction system can't just output one trajectory; it needs to capture the full range of plausible futures. Previous approaches use GANs or CVAEs to model this multi-modality, but GANs are unstable to train and CVAEs tend to produce unrealistic trajectories. This paper takes a different approach: it treats trajectory prediction as a denoising diffusion process, starting from random noise (representing all possible walkable areas) and gradually refining it into a realistic trajectory. The model observes 3.2 seconds of past motion (8 timesteps) and predicts the next 4.8 seconds (12 timesteps).

## 2. Assumptions and Hypotheses

The key assumption is that generating a trajectory can be modeled as the reverse of a noise-adding process. If we gradually corrupt a real trajectory with Gaussian noise until it becomes pure randomness, then a neural network can learn to reverse those steps — going from noise back to a plausible trajectory. The paper also assumes that social context matters: a pedestrian's future path depends on nearby people (anyone within 3 meters), so the model encodes neighbor interactions through a spatial-temporal graph. The six input features per pedestrian — position, velocity, and acceleration in x and y — are assumed to capture enough motion information without needing scene images or maps. Finally, the diffusion chain length (100 steps) controls a tradeoff: longer chains produce more accurate but less diverse predictions, and shorter chains give more variety at the cost of precision.

## 3. Exploratory Data Analysis

The exploratory analysis covers all five ETH/UCY scenes (ETH, Hotel, Univ, Zara1, Zara2) along three axes: dataset scale, motion statistics, and social structure. Scene size and crowd density vary substantially — Univ is by far the densest while ETH and Hotel are markedly sparser — and trajectory length distributions are heavy-tailed, with many pedestrians appearing only briefly at frame edges and only a fraction reaching the 20 frames required for the standard 8+12 history/future split. This directly constrains the amount of usable training data per scene. Pedestrian speeds cluster tightly around walking pace (roughly 1.0–1.5 m/s) with a long tail of near-stationary observations, meaning the model must handle both smooth motion and quasi-static segments. Position heatmaps reveal strong scene-specific spatial priors at doorways, sidewalk edges, and crossing points. Crucially, nearest-neighbor distance distributions validate the 3-meter attention radius used downstream: a meaningful fraction of pedestrians have at least one neighbor within 3 m at any given timestep, especially in Univ and the Zara scenes, confirming that neighbor interaction is a frequently-active signal rather than an edge case. These observations motivate two hypotheses for the modeling side: social conditioning should yield larger gains on dense scenes than on sparse ones, and cross-scene performance differences are likely driven as much by 8+12 data sparsity as by intrinsic scene difficulty.

## 4. Feature Engineering and Data Transformations

The raw ETH/UCY data is just frame IDs and (x, y) positions for each pedestrian. The preprocessing pipeline turns this into training-ready samples through several steps. First, velocity and acceleration are computed from position using central differences, giving six features per timestep instead of two. All features are standardized — positions are divided by the attention radius (3.0 meters), velocities by a standard deviation of 2, and accelerations by 1 — and made relative to the pedestrian's current position so the model sees displacement rather than absolute coordinates. A scene graph is constructed at each timestep by checking pairwise distances between all pedestrians, with temporal smoothing filters applied so that neighbor relationships don't flicker on and off. For training, each scene is augmented with 24 rotated copies (every 15 degrees) to make the model rotation-invariant. The ETH test set also gets a 0.6x coordinate scaling correction to match the training coordinate system. The final output is serialized `.pkl` files containing Environment objects with Scene, Node, and SceneGraph structures that the Trajectron++ encoder expects.

## 5. Proposed Approaches

*To be completed after running experiments.*

## 6. Model Selection and Architecture

The MID framework has two parts: a Trajectron++ encoder and a Transformer-based diffusion decoder. The encoder is borrowed from prior work — it uses an LSTM to encode each pedestrian's history and an attention mechanism to aggregate information from nearby neighbors, producing a 256-dimensional state embedding. The diffusion decoder is the paper's contribution. It uses a three-layer Transformer (512 dimensions, 4 attention heads, feedforward size 1024) with a gating mechanism called ConcatSquashLinear, where each layer's output is multiplied by a learned gate and shifted by a learned bias, both conditioned on the diffusion timestep and the encoder's state embedding. The noise schedule is linear from 1e-4 to 5e-2 over 100 steps. The model is trained with Adam (lr=0.001, batch size 256) for 90 epochs. Regularization comes from several sources: dropout in the Transformer, the stochastic nature of the diffusion process itself, and the decision to use only trajectory features without scene images, which prevents overfitting to visual cues. The 512-dimensional Transformer was selected over 256d and 1024d variants based on ablation results, and the diffusion formulation was validated against a CVAE baseline using the same encoder and decoder architecture.

## 7. Results

*To be completed after running experiments.*

## 8. Future Work

The main limitation of MID is inference speed — generating one prediction requires 100 sequential denoising steps, making it roughly 40 times slower than Trajectron++ (17.4 seconds vs 0.4 seconds for 512 trajectories). Recent fast sampling methods like DDIM could reduce this to 10-20 steps with minimal quality loss, but integrating them is nontrivial. Another direction is incorporating scene images or maps as additional conditioning, since the current model uses only trajectory data and ignores physical obstacles like walls or curbs. The model could also be tested with adaptive chain lengths — using fewer diffusion steps in simple straight-line scenarios and more steps in crowded or ambiguous situations. Finally, the framework could extend beyond pedestrians to multi-agent settings with vehicles and cyclists, which the Stanford Drone dataset already provides but MID doesn't use.

## References

- Gu, T., Chen, G., Li, J., Lin, C., Rao, Y., Zhou, J., & Lu, J. (2022). Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion. CVPR.
- Salzmann, T., Ivanovic, B., Chakravarty, P., & Pavone, M. (2020). Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data. ECCV.
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
