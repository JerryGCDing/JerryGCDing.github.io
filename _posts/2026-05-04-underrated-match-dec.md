---
layout: distill
title: The Underrated Decoder Behind a Unified Correspondence Model
description: On attention-as-matching-cost and a path to scaling correspondence
tags: computer-vision algorithm
categories: reflection
date: 2026-05-04 10:00:00
featured: true
thumbnail: /assets/img/blog/geo-matching/two_stream_idea.png

toc:
  - name: Introduction
  - name: Previous Works in Visual Correspondence
  - name: Attention and Matching
  - name: Dual-Stream Cross Attention
  - name: Matching Decoder Layer and Large Correspondence Model
  - name: Conclusion
---

## Introduction
 
3D computer vision is in the middle of a quiet consolidation. Tasks that used to live in their own walled gardens — depth estimation, structure from motion, point cloud registration — are increasingly being absorbed into a small number of large, Transformer-based foundation models like
[DUSt3R](https://arxiv.org/abs/2312.14132), [MASt3R](https://arxiv.org/abs/2406.09756), and [VGGT](https://arxiv.org/abs/2503.11651). 
The pattern is the same one that played out in NLP a few years earlier: a single architecture, scaled with diverse data, ends up generalizing better than a zoo of specialists.
 
There is one task sitting at the very foundation of 3D vision that has stubbornly resisted this
consolidation: **visual correspondence**. The job of finding the same physical point across two different observations is the substrate that registration, camera pose estimation, SLAM, and SfM all stand on. 
And it shows up in three flavors — image-to-image (2D-2D), image-to-point cloud (2D-3D), and point cloud-to-point cloud (3D-3D) — depending on what modality the source and target observations come in. 
Despite all three being the same problem at heart, *given a keypoint here, find its match over there*, the field still trains a separate specialist for each.
 
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/blog/geo-matching/unicorrn_teaser.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    UniCorrn finds correspondences across 2D-2D, 2D-3D, and 3D-3D pairs with a single set of weights.
</div>

In our recent work [*UniCorrn*](https://arxiv.org/abs/2605.04044), we asked whether one model can do all three. 
The short answer is yes — a single 600M-parameter model with shared weights matches the leading
2D-2D specialists and beats the leading 2D-3D and 3D-3D specialists by 8% and 10% in registration recall on 7Scenes and 3DLoMatch, respectively. 
The longer answer, which is what I want to write about here, is that getting there required committing to a **matching decoder** design that I think is badly under-explored relative to its potential. 
This post is about that decoder — the design choices that make it work, why the matching decoder is the underrated piece in current correspondence foundation models, and why I think this kind of design is the right substrate to bet on for a future Large Correspondence Model. 
I want to walk through the reasoning roughly the way I worked through it, because each design move only really makes sense in light of the constraint the previous one created.
 
Underneath the technical specifics is a research instinct I keep returning to: that the most interesting wins in modern computer vision sit at the intersection of classical structural priors and large neural architectures, and that scaling carries you furthest when it is paired with the right explicit inductive bias rather than asked to do everything alone. 
UniCorrn is one such pairing. This post is about how it was built.

## Previous Works in Visual Correspondence

Before getting to the design space I want to defend, it is worth laying out what people have been doing instead. Existing approaches to learning visual correspondence — including the ones that have tried to unify across multiple settings — fall into roughly four families.
 
**Cost volume and dense warp refinement.** 
[RAFT](https://arxiv.org/abs/2003.12039) is the canonical case for optical flow: build a 4D correlation volume between feature maps and refine the flow estimate iteratively through a recurrent unit. 
For wide-baseline matching, [DKM](https://arxiv.org/abs/2202.00667) recasts matching as kernel regression and refines a dense warp through stacked feature maps; [RoMa](https://arxiv.org/abs/2305.15404) extends this lineage with DINOv2 features, an anchor-probability match decoder, and ConvNet refiners at multiple resolutions. 
These methods currently top the 2D-2D leaderboards. 
The architectural cost of being accurate is rigidity: the pyramid depth is fixed, the recurrence has finite unrolling, and the underlying machinery — correlation volumes, warp grids — assumes a 2D pixel grid that does not extend gracefully to irregular 3D point sets.
 
**Dense descriptor matching with a single-stage matching layer.**
[LoFTR](https://arxiv.org/abs/2104.00680) learns dense descriptors with a Transformer and matches them through a dual-softmax mutual-matching layer at coarse resolution, refined locally at the fine level.
[MASt3R](https://arxiv.org/abs/2406.09756) augments DUSt3R with a dense feature head and uses fast
reciprocal nearest-neighbor matching at inference. 
The descriptor and the matcher are tightly coupled, but the matching itself is a single stage at the end of a feature backbone — you cannot stack matching layers and refine through them the way Transformer decoders are designed to work, which means the descriptor has to do all the heavy lifting in a single forward pass.
 
**Direct regression.** 
[UFM](https://arxiv.org/abs/2506.09278) is the cleanest current example: fuse the two image features with a generic Transformer encoder and directly regress the dense pixel displacement
field. 
Fast, and it scales beautifully on 2D-2D. 
But our experiments show it falls over on 2D-3D and 3D-3D — without an explicit similarity mechanism in the architecture, the model has nowhere to encode geometric reasoning about *which target token actually corresponds to my source query*.
 
**Sequence concatenation.** 
[COTR](https://arxiv.org/abs/2103.14167) concatenates source and target into a single sequence and lets a vanilla Transformer figure out matching. 
Flexible, but in our ablations (Table 1 of the paper) it performs the worst across all three settings, presumably because the model has to discover the matching structure from scratch with no built-in bias.
 
What I want to argue is that the four families above all share a quiet assumption: the *matching decoder* is a small, low-investment afterthought sitting on top of a large feature backbone. Nearest neighbor is one line. 
Regression heads are an MLP. 
Cost volumes do their work in a fixed correlation block before the network really starts thinking. The actually interesting design space sits in a matching decoder that hits all three desiderata at once — (1) end-to-end learning through stackable layers, (2) handling of irregular structures like point clouds without requiring a 2D grid, and (3) iterative geometric refinement of the correspondence estimate — and that space has been under-explored. 
The point of this post is what such a decoder looks like, and why I think building it right is the most consequential architectural choice in a unified correspondence model.

## Attention and Matching
 
The starting observation is one that has been hiding in plain sight inside every Transformer ever written.
 
Take two sets of tokens, source $$F_s$$ from one observation and target $$F_t$$ from another, and write down a single cross-attention layer. 
Compute the (row-normalized) attention matrix $$A$$ between source queries and target keys:
 
$$A = \mathrm{Softmax}\left(\frac{F_s W_Q (F_t W_K)^\top}{\sqrt{D}}\right).$$
 
In an ideal world where features are perfectly discriminative, each row of $$A$$ is a one-hot vector — and
the position of the 1 in row $$i$$ is exactly the index of the target token that matches source token $$i$$.
*That is a matching cost.* Up to row-normalization, $$A$$ is the same object as a learnable cost volume
(see [Xiao et al. 2020](https://arxiv.org/abs/2007.11431)).
 
Now do something slightly unusual with the value vector $$V$$. Instead of feeding it the appearance
features of the target tokens, feed it the *absolute positional encoding of every target token*. The output of the attention layer is then
 
$$Q = AV,$$
 
which, for each source query, returns the positional encoding of its corresponding target location. 
Push that through a linear layer and you have *coordinates*. The supplementary material of our paper has a tiny toy illustration of this with four pixels and four animal symbols that I think is genuinely the most useful one-figure summary of why the rest of the model looks the way it does.
 
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/blog/geo-matching/two_stream_idea.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Attention as a learnable matching cost. 
    With perfect features, the attention matrix becomes a permutation matrix; multiplied by target positional encodings, it reads off the matching coordinates directly.
</div>

This is the seed of what I think of as **geometry-based matching with attention**. It treats the attention matrix as the matching cost (geometric), and it produces matching *coordinates* (also geometric) rather than just matching descriptors. 
It is also automatically modality-agnostic: nothing about the construction assumes the target tokens come from a 2D grid, they could just as easily be 3D points.
 
But there is an immediate problem. **What happens when you try to stack two of these layers on top of each other?**

## Dual-Stream Cross Attention
 
The reason single-layer cost-volume reasoning is not enough is that visual correspondence is hard. With real features, $$A$$ is never a perfect permutation matrix; it is a soft, noisy approximation. 
The whole point of Transformer-style stacking is that we want layer 2 to look at the rough estimate from layer 1 and refine it. 
That requires layer 2 to compute its *own* attention matrix between source and target — which means it needs appearance features for the source query.
 
And there is the rub: if layer 1 sets $$V$$ to positional encoding to read off coordinates cleanly, the output is positional only — there are no appearance features for layer 2 to query with, and the design is fundamentally non-stackable. 
The alternative, keeping $$V$$ as appearance features (the standard recipe used in basic regression-style decoders), does stack, but at the cost of blending: the prediction head now has to disentangle matching coordinates from a representation where positional and appearance signals are mixed together in the shared channels. 
The single-stream design forces a choice between these two failure modes.

Getting there took a couple of detours. 
The earliest small-scale experiments — regression-style and
sequence-concatenation baselines — failed by large margins, working only at large scale and only on
2D-2D. 
My first structural fix was to store positional features in a single hidden state and update them
through a gated operator at each layer. That ran into exactly the failure mode above: positional features kept colliding with the appearance content the query was carrying, and stacking layers did not help. 
The disentangled, two-stream version suggested itself when I went back to the basics of how attention is supposed to work — queries and values are different things, and there is no rule against giving them their own residual streams when the values you want to read out are categorically different from the features you want to attend with.
 
This is the gap the **dual-stream Transformer decoder** fills — the central design move in the paper, and the one I want to walk through carefully, because it is the move the rest of the decoder is designed around.

The idea is to maintain *two parallel residual streams* through the decoder — one for appearance, one for position — and have them share a single attention matrix. 
Concretely, for each source keypoint we carry along both an appearance feature $$F_k$$ and a positional embedding $$P_k$$, the latter initialized to zero since we do not yet know where the match is. 
At every layer:

1. Compute the attention matrix $$A$$ from $$F_k$$ and $$F_t$$, with both streams contributing position information through RoPE.
2. Update the appearance stream: $$F_k \leftarrow A \cdot W_V F_t + F_k$$. This is the standard Transformer update.
3. Update the positional stream: $$P_k \leftarrow A \cdot \mathrm{AbsPE}(X_t) + P_k$$. The same attention matrix, but applied to the *target tokens' absolute positional encoding*.

Each stream has its own residual connection, so both quantities flow through the stack and are gradually refined. 
Crucially, the next layer recomputes $$A$$ from the updated $$F_k$$, so the attention itself
improves layer over layer; and because $$P_k$$ accumulates "weighted target positions" along the way, by the final layer it encodes the regressed coordinate of the corresponding keypoint. 
A single linear head pulls the coordinates out at the end via the Moore–Penrose inverse of the AbsPE map.
 
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/blog/geo-matching/dual_stream_decoder.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Dual-stream attention with a single shared attention matrix. 
    Appearance features and positional embeddings travel in parallel residual streams; after attention, the joint output is split and each stream is updated independently.
</div>

A few practical details I want to highlight, because each was a step on the path between the prototype and what is in the paper now.
 
**Positional encoding: from sinusoidal to RoPE.** 
In my initial prototype of the dual-stream layer I used additive sinusoidal positional encoding, applied directly to $$Q$$ and $$K$$ just before the attention computation rather than baked into the input embeddings (the vanilla pattern). 
It worked well on 2D-3D and 3D-3D matching — settings where the target modality is a point cloud, but broke on 2D-2D. 
It led us to switch to [**RoPE**](https://arxiv.org/abs/2104.09864), which rotates $$Q$$ and $$K$$ multiplicatively so the inner product itself depends on relative position rather than on a perturbed additive sum, fixed it across the board. 
The practical lesson, in retrospect, is that for matching you want positional information to inform the *geometry* of the attention computation, not sit on top of it as an additive perturbation.

**A decodable positional encoding for the value stream.** 
There is a parallel choice on the value side. The positional stream is updated as $$P_k \leftarrow A \cdot \mathrm{AbsPE}(X_t) + P_k$$, and the final coordinates come out by inverting AbsPE. 
My first version used a learnable sinusoidal Positional Embedding (PE) — sinusoids passed through an MLP projection — but the decode at the end was an approximation rather than exact, because the MLP non-linearity does not invert cleanly.
Once I started thinking of this stream as *a value that has to be decodable back to a coordinate*, the constraint became obvious: AbsPE has to behave arithmetically like a raw coordinate, so that $$\mathrm{AbsPE}(A) + \mathrm{AbsPE}(B) = \mathrm{AbsPE}(A + B)$$. 
A learned affine map $$W_p X + b_p$$ satisfies that exactly, and its Moore–Penrose pseudo-inverse $$W_p^+$$ gives a closed-form decode. 
The deeper point is that linearity is what preserves the global-matching structure end-to-end: the attention-weighted average $$A \cdot \mathrm{AbsPE}(X_t)$$ stays equal to AbsPE applied to the regressed target coordinate, instead of drifting through a non-linear distortion.
 
**Gaussian attention instead of vanilla dot-product.** 
This one started somewhere unexpected. 
Optimal transport (OT) is a long-standing tool for matching and registration — Sinkhorn-style soft matching is widely used across the point-cloud and image-matching literature — and my initial plan was to incorporate an OT-style learnable module into the matching decoder. 
As I worked through the formulation, the math kept reducing to plain attention with a Gaussian kernel: one entropic-OT iteration with squared-$$L^2$$ cost is exactly

$$A = \mathrm{Softmax}\left(-\frac{\|F_k' - F_t'\|^2}{D}\right).$$

So I kept the form and dropped the framing. 
In the paper we present it through the cleaner descriptor-matching lens — Gaussian kernels are what classical descriptor matching has always preferred, capturing non-linear similarity and robust to feature magnitude — but the design move was originally derived from OT, and either framing produces the same kernel. 
Empirically a free win over vanilla dot-product attention, and conceptually it is what the matching framing tells you to use.
 
**Single-head attention.** 
A counterintuitive ablation: since each query has at most one corresponding target token, the matching problem is closer to nearest-neighbor than to multi-relational reasoning.
Reducing the number of attention heads from 16 to 1 in the matching decoder consistently improved
performance. 
Multi-head attention in this layer was, in retrospect, an artifact of importing the encoder template uncritically.

## Matching Decoder Layer and Large Correspondence Model
 
Once the dual-stream layer is the right shape, two more ingredients turn it from an "interesting decoder block" into "the matching decoder you would want in a large correspondence model."
 
**Auxiliary supervision through the attention itself.** 
The dual-stream design buys you stackability, but it does not *use* the stackability for free. The fix came from two threads converging. 
First, recognizing that attention-as-matching-cost is itself a form of global matching, we ran that observation as a baseline — a single-layer global matcher that reads out coordinates as $$A \cdot X_t$$. It works, but only at full feature-map resolution, where the cost is too high to be practical (the "global matching" row in Table 1 of the paper). 
Second, while the dual-stream decoder with RoPE was already converging well, the intermediate attention maps I inspected were diffuse and not consistently focused on the matching coordinate — the kind of unfocused mid-stack attention Transformer decoders fall into when nothing supervises them.
Folding the two together gave the fix: every intermediate layer already produces an attention matrix $$A^{(l)}$$ that is, by construction, a soft correspondence map, so apply the same global-matching read-out at each layer as auxiliary supervision. 
Read out a coordinate estimate $$K_t^{(l)} = A^{(l)} X_t$$, supervise it with the same L1 matching loss used at the output, and the diffuse intermediate attention sharpens. 
The auxiliary loss is
 
$$\mathcal{L}_{\mathrm{aux}} = \sum_{l=1}^{L} \gamma^{L-l} \cdot \frac{1}{N} \sum_{i=1}^{N} \left\| K_t^{(l)}(i) - \bar{K}_t(i) \right\|_1.$$
 
This is the design choice that turns out to have the most empirical weight in the whole decoder, and the one most easily missed. 
Without it, the per-layer attention heatmaps look essentially random — the layers are ostensibly
being trained, but they are not learning to represent anything coherent on their own. 
With it, every intermediate layer produces a sharp attention map focused on the predicted coordinate, and the layers behave like an honest iterative refinement of the matching estimate. The supplementary visualization makes this dramatically clear:
 
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/blog/geo-matching/aux_heatmap_compare.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Per-layer attention heatmaps without (left) and with (right) auxiliary supervision, green markers indicate the model's predicted coordinates. 
    Without it, intermediate layers are diffuse and noisy. With it, each layer attends coherently to the predicted correspondence.
</div>

The numbers tell the same story. With a single matching decoder layer, adding the auxiliary loss takes AUC@5° on MegaDepth-1500 from 28.8 to 47.7; with five layers, it still adds 2 points on top of an already
strong setup. 
The implication, which I think is the more important lesson, is that the dual-stream design *needs* layer-wise supervision to actually behave like a multi-layer refiner, rather than an opaque end-to-end function whose intermediate layers do whatever.
 
**A contrastive descriptor loss.** 
On top of the matching loss, we apply an InfoNCE term between the source and target feature descriptors at ground-truth correspondence locations, following [MASt3R](https://arxiv.org/abs/2406.09756)'s descriptor-supervision recipe. 
This sharpens the underlying feature similarity that the attention mechanism feeds on. 
Not a novel piece in this paper, but the descriptor and the matching mechanism are entangled, and supervising them jointly works better than supervising matching alone.
 
**Putting it together: the Large Correspondence Model.** 
Stack 8 dual-stream decoder layers on top of a shared cross-attention encoder, sit that on top of modality-specific backbones (a ViT for images, [Point Transformer v3](https://arxiv.org/abs/2312.10035) for point clouds), and you have UniCorrn — 600M parameters, one set of decoder weights, three correspondence tasks. 
Trained jointly on a mix of 2D-2D image pairs, 2D-3D and 3D-3D real correspondence data, and pseudo-3D data lifted from depth maps, the model:
 
- matches the leading 2D-2D specialists on MegaDepth-1500 and ScanNet-1500;
- beats the previous SOTA 2D-3D specialist by **8% in registration recall** on 7Scenes;
- beats the previous SOTA 3D-3D specialist by **10% in registration recall** on the harder 3DLoMatch
  benchmark;
- and does **zero-shot optical flow on Sintel at 5.2 EPE**, despite never being trained on dynamic or stylized imagery. 
That is the kind of out-of-distribution generalization signal you would expect from a foundation model — and it is what makes me believe this decoder design is the one that scales. The single unified model also uses 3.5× less memory than the combined memory footprint of separate
specialists for the three tasks, which matters quite a bit in deployment.
 
There is one honest negative result worth surfacing too: joint training on all three tasks does not always beat joint training on two. 
A gradient-conflict analysis traced this to the normalization layers, which struggle to share statistics across the very different distributions of 2D image features and 3D point features. This is a real bottleneck and a clear next step for anyone trying to push this paradigm further.

## Conclusion
 
There are a lot of moving parts in UniCorrn, but the thesis underneath is small. 
A unified, scalable correspondence model needs three things: stackable layers, modality-agnostic structure, and an explicit geometric matching mechanism. 
The paradigms that currently dominate each give up at least one of these.**Attention-as-matching-cost** gives you all three for free, *if* you are willing to do the architectural work of (a) decoupling appearance and position into two streams so layers can stack, (b) supervising every layer through its own attention to make those stacks behave, and (c) regressing actual coordinates instead of just descriptors so the output is geometric rather than featural.
 
I think this is the underrated piece. 
Foundation work on correspondence has been pouring its budget into encoders — bigger backbones, richer feature fusion, more pretraining — and treating the matching decoder as plumbing. 
But the decoder is where the inductive bias for matching actually lives, and the shape of that decoder determines what the model can do. 
Cost volumes are pinned to the grid. Nearest neighbor is a single shot. Direct regression is brittle without a similarity prior. 
A decoder that uses attention as a stackable, iteratively refined matching cost is, in my view, the one with the right architectural *shape* for the unified correspondence problem — and if a Large Correspondence Model that handles every flavor of correspondence (geometric, semantic, temporal) under a single set of weights does emerge, this is the kind of decoder I would expect to be at its core.
 
What this work demonstrates and what it merely suggests are different things, and I want to be honest about which is which. 
UniCorrn establishes that this decoder design is *better* than the alternatives at 600M parameters across three correspondence tasks. 
It does not yet establish that the design *scales* in the strong sense the foundation-model framing implies. 
The decoder-depth ablation plateaus around three layers in the small-scale ablation, and the large-scale model only goes to eight; there is no scaling curve past that yet. 
Joint training surfaces real gradient conflicts in the normalization layers as the modality count grows, and that does not get easier as semantic and temporal modalities are added.
Pseudo-3D data carries a lot of weight in the current results — without it, performance on 7Scenes drops from 77.8 to 15.4 RR. None of these are dealbreakers, but they are the open questions the next round of work has to answer, and I think they are more interesting than the claim of state-of-the-art numbers in a single paper.
 
What I am betting on — and what this post is trying to articulate rather than prove — is the broader instinct underneath the work, the one I keep returning to in everything I do: scaling carries you a long way, and the rest of the way is reached by the right explicit structural inductive bias. 
Geometry, compositionality, the math of the problem itself. 
Attention-as-matching-cost is the bias that fits correspondence; what I think is genuinely interesting is finding the others, for the parts of 3D and 4D vision that scaling alone has not yet reached, and figuring out what makes a correspondence model *genuinely* scalable rather than just bigger.