# Fig. 2: dynSVD router weights

plot_router_weights.py evaluates the trained dynSVD router on five evaluation
questions and plots its softmax weight over the first five SVD residual
vectors. Each checkpoint passed to --checkpoints is treated as one dataset.
Five checkpoints therefore produce a 5 (questions) x 5 (datasets) grid; one
checkpoint produces five question panels.

The router input is the normalized hidden state at the final token of the
Q: <question> A: prompt, matching model/methods/dynSVD/evaluate_dynSVD.py.
The script does not train or run MC scoring.

Example from the repository root:

    /root/miniconda3/envs/py310/bin/python \
      experiments/fig2/plot_router_weights.py \
      --checkpoints \
        features/router_dataset1.npz \
        features/router_dataset2.npz \
        features/router_dataset3.npz \
        features/router_dataset4.npz \
        features/router_dataset5.npz \
      --model_name llama3.1-8B-Instruct \
      --model_dir /mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct \
      --num_questions 5 \
      --num_vectors 5 \
      --output_prefix experiments/fig2/figures/fig2_router_weights

For the current fold checkpoints:

    /root/miniconda3/envs/py310/bin/python \
      experiments/fig2/plot_router_weights.py \
      --checkpoints \
        features/llama3.1-8B-Instruct_layer14_dynSVD_fold0.npz \
        features/llama3.1-8B-Instruct_layer14_dynSVD_fold1.npz \
      --model_dir /mnt/nas-public-bing/pinxuanli11/Meta-Llama-3.1-8B-Instruct

Outputs are written next to the chosen prefix: PNG/PDF/SVG figures, a
long-form CSV, and a grouped JSON table. Use
--question_ids 12 34 56 78 90 to select exact questions. If omitted, the first
five sorted test_q_indices in the first checkpoint are used.
