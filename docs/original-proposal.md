# Training a Neural Network with Jev Neurons

Research and experimental design

26 September 2026

> Pre-audit version, kept for history. The current plan is [proposal.md](proposal.md).

We propose training an image classifier whose neuron activations are supplied by Jev API evaluations. The optimizer learns the outer network's weights and biases while Jev remains frozen. The main research question is whether those learned connections can compensate for an imperfect model acting as a neuron.

The first implementation should use Jev as a scalar activation function. A second experiment should send the complete neuron state, including inputs, weights, and bias, so Jev performs the weighted-sum decision itself. Both aim at the same visible result: a network of Jev evaluations learning to recognize handwritten digits.

## The starting point

The precursor is Raj's Jev NAND-gate experiment: a 4-bit ALU reportedly computes 7 + 5 = 12 using 116 gates in 7.6 seconds for $0.0018, with a reported compound probability of 84%. These are the supplied demo measurements. The next experiment adds learning: connections change in response to labeled examples.

The reported compound probability should not be treated as measured end-to-end correctness without validating its assumptions. For the classifier, report accuracy on held-out images directly; errors from the same underlying model can be correlated.

## Proposed first classifier

```text
784 pixel inputs (28 × 28 image)
  → learned weights → 32 Jev hidden neurons
  → learned weights → 10 Jev digit outputs

25,450 learned weights and biases
One shared frozen Jev model
```

Start with a 784 to 32 to 10 candidate using native 28 by 28 MNIST images. It has 42 evaluated neurons and 25,450 trainable parameters. Validate its capacity locally before spending on Jev training. The same frozen Jev model serves every neuron; separate weights and input states define their roles.

This is a proposed experiment. Convergence, useful accuracy, runtime, and cost have not been measured. Existing work supports the optimization methods described here; it does not establish that Jev can be trained into this classifier.

## Choosing the network size

The recommended first digit candidate is a fully connected 784 to 32 to 10 network. The 784 inputs are pixel values, the 32 hidden units learn intermediate features, and the ten outputs correspond to digits. Input pixels do not invoke Jev; the 42 evaluated units are reused for every image.

The earlier 64 to 16 to 8 to 10 proposal was a cheap pilot using images resized to 8 by 8. Its 34 units were a budget choice, not a standard MNIST architecture or a measured minimum. Network size remains an experimental choice. Standard training learns weights and biases from labeled examples; using a remote frozen model as an activation and approximating its gradients is the unusual part. [16](https://cs231n.github.io/neural-networks-1/)

| **Candidate** | **Evaluations per image** | **Trainable parameters** |
| --- | --- | --- |
| 64 → 16 → 8 → 10 compact pilot | 34 | 1,266 |
| 784 → 32 → 10 first candidate | 42 | 25,450 |
| 784 → 64 → 10 wider candidate | 74 | 50,890 |
| 784 → 128 → 10 tutorial sized candidate | 138 | 101,770 |

### Keep the original pixels for scalar neurons

In Variant A, code computes the weighted sum and sends Jev only one scalar. Keeping 784 pixels instead of 64 increases local parameters, but does not increase the number of Jev evaluations for a fixed hidden width. Exact token use still depends on number formatting and batching. In Variant B, supplying 784 input-weight pairs makes states longer and arithmetic harder, so reduced-resolution or shorter-vector experiments remain useful.

### Check capacity before testing Jev

Compare conventional networks with hidden widths 32, 64, and 128 on the same training and validation split. Define a useful demo accuracy target before paid training, and select the smallest candidate that meets it consistently across seeds. Do not assume 42 units are enough merely because the architecture is affordable. Keep the test set untouched while selecting width.

TensorFlow's introductory MNIST tutorial uses 784 inputs, 128 ReLU hidden units, dropout, and ten output logits, and reports about 98 percent test accuracy. That is a conventional reference, not an accuracy forecast for Jev or a guarantee for a 1,000-image pilot. [15](https://www.tensorflow.org/tutorials/quickstart/beginner)

After selecting width, add a local control with output activations and loss matched to the Jev design. Compare ordinary training, substituting Jev after training, and training with real Jev forward passes. A weak conventional baseline points to capacity, data, or optimization; a strong baseline followed by a weak Jev run points toward the replacement or its backward rule.

### Decisions for review

Review the scalar-first design, the progression from truth tables to digits, the validation rule for choosing width, and the proposed spending caps. The cost tables use the 42-unit candidate so the assumptions are explicit. A different width requires a new estimate, and no architecture or accuracy result is presented as experimentally established.

## What the optimizer learns

Optimizing Jev's input state means updating persistent numeric parameters that are serialized into each call. It does not mean changing the training images or fitting a separate input for every example. The same learned parameters must classify unseen images.

| **Component** | **Changes during training** | **Role** |
| --- | --- | --- |
| Connection weights and bias | Yes | Persistent parameters for each outer neuron |
| Pixel values and incoming activations | Per example | Data from the image or previous layer |
| Jev internal parameters | No | Shared frozen evaluator |
| Instructions and criteria | Initially fixed | Define the neuron operation |

### Variant A uses Jev as the activation

Code computes the weighted sum, then asks Jev whether that single number is positive. The returned yes probability becomes the activation. A Noul value is a probability about the stated proposition; here we deliberately use it as a numeric signal. It is not a derivative or a measurement of the number's magnitude. The snippets are pseudocode. [1](https://docs.typesafe.ai/primitives/noul)

```text
z = dot(weights, incoming_activations) + bias
state = {"z": z}
activation = Jev(state, question="Is z greater than zero?")
```

The unknown function has one input, z. This makes it possible to measure an activation curve and estimate its slope. The weights live in the local training code; their updates change the state sent to Jev. Define zero explicitly as a no and use one fixed serialization format.

### Variant B sends the whole neuron state

```json
{
  "inputs": [0.2, 0.8, 0.1],
  "weights": [0.4, -0.7, 0.9],
  "bias": 0.15
}
```

The fixed question is: Is the sum of inputs[i] times weights[i], plus bias, greater than zero? Training updates the supplied weights and bias. Jev interprets the vector, performs the arithmetic, and returns the activation. This is the more literal model-as-neuron experiment, with a harder optimization problem.

TypeSafe documents numeric precision and arithmetic weaknesses for Jev 1.13. Those limitations make a scalar experiment a useful starting point, and make the whole-neuron experiment a substantive test of adaptation. The limitation is version-specific and should be measured again for the model used. [7](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

### Output semantics

Use ten independent output activations with one-hot digit targets. Train with mean binary cross-entropy across the ten outputs, clipping values only for numerical stability, and predict the largest output. These ten Noul values need not sum to one and are not automatically calibrated digit probabilities. Keep the output-to-digit mapping in code.

## Representing and storing neuron state

State has two meanings in this experiment. The training system stores persistent learned parameters. Each Jev request then presents a view of those parameters and the current example. TypeSafe accepts text and structured JSON as request state; digit images must be converted into numeric inputs in code. [11](https://docs.typesafe.ai/concepts/state)

### Keep the canonical state in the training system

Store weights and biases as numeric arrays in a checkpoint, with a JSON manifest for the architecture, prompts, schema, model version, and preprocessing. Save optimizer state and random seeds so a run can resume. Preserve full numerical precision in the checkpoint even when requests use rounded values.

For each forward pass, construct fresh request state from that checkpoint and the current activations. Treat per-example activations as temporary data. Each neuron gets its identity from its learned parameters and position in the graph; it does not need a persistent conversation or a separate Jev model instance.

### Compare different divisions of the computation

| **Representation** | **State presented to Jev** | **Work performed by Jev** |
| --- | --- | --- |
| Scalar activation | One weighted sum z | Threshold judgment |
| Precomputed contributions | Products w_i × input_i and bias | Summation and threshold judgment |
| Complete neuron | Inputs, weights, and bias | Multiplication, summation, and threshold judgment |
| Semantic neuron extension | Named features and detection criteria | A learned or selected concept judgment |

Precomputed contributions provide a useful intermediate experiment: they isolate summation errors from multiplication errors. Semantic neurons change the hypothesis being tested and may require prompt optimization; keep them separate from the first numerical classifier.

### Compare encodings while holding the task fixed

For complete states, compare parallel arrays with paired records such as {"input": 0.2, "weight": 0.4}. Paired records make the relationship explicit but cost more tokens. A compact English rendering of the same values is another format to test. Keep the data, ordering, question, and evaluation examples identical across these comparisons.

Choose a fixed decimal precision per run. Scaled integers are another representation, but define every scale explicitly: multiplying two scaled values changes the units, and bias must use matching units. Coarse rounding changes the effective model and can erase small optimizer updates. Treat binary weights or coarse quantization as separate experiments, not equivalent formatting.

Store the exact transmitted state and returned probability for diagnostic samples. Cache only exact matches of model version, prompt, criteria, and serialized state, and label cached runs separately. A repeated call may vary, so a cache can change the behavior of a live-call experiment.

## Using the probability output

The recommended first network preserves Jev's yes probability throughout its hidden layers. If a neuron returns 0.83, send 0.83 to the next layer. Turning it into 1 immediately discards differences between answers such as 0.51 and 0.99. This use treats the probability as a signal; it does not require interpreting it as calibrated classification confidence. [1](https://docs.typesafe.ai/primitives/noul)

### Compare soft and binary activations

```text
soft_activation = p
signed_activation = 2 * p - 1
hard_activation = 1 if p >= 0.5 else 0
sampled_activation = 1 if uniform_random_0_1() < p else 0
```

Start with the soft activation. Compare the signed mapping in hidden layers to test whether centering the signals helps optimization; keep output activations in [0, 1] for the losses below. Hard thresholding provides a binary control. Sampling locally creates a Bernoulli neuron whose firing probability is p, without requiring an extra Jev call for that sample.

A sampled hidden state changes the inputs to downstream neurons, which must be evaluated for that state. Compare one stochastic pass with averages over several complete passes, and count all evaluations. Soft propagation need not equal the average prediction of a sampled network because subsequent layers are nonlinear. [12](https://arxiv.org/abs/1506.05254)

### Use probabilities in the loss

For a target y equal to zero or one, compare binary cross-entropy with squared probability error, also called Brier loss:

```text
BCE(p, y) = -y * log(p) - (1 - y) * log(1 - p)
Brier(p, y) = (p - y) ** 2
```

Both respond to changes in p even when it remains on the same side of 0.5. Cross-entropy penalizes confident mistakes more sharply; squared error is bounded. Clip values for logarithms and specify the backward rule at the boundaries. For ten digit outputs, average the chosen per-output loss against one-hot targets and use argmax for classification.

### Use the distribution to inspect the network

Render p as neuron brightness and track saturation near zero and one. Bernoulli variance p × (1 - p), or binary entropy, describes how spread out a sampled neuron would be. Measure repeated-call variability separately. These diagnostics do not establish how often the neuron or classifier is correct.

A later experiment could allocate more stochastic passes to ambiguous predictions. Choose the stopping policy on validation data and measure its accuracy versus cost. Extra sampling can stabilize an estimate but does not guarantee better accuracy.

### Probability is still not a derivative

The value p × (1 - p) is the derivative of a logistic sigmoid with respect to its logit. It is not automatically the derivative of Jev with respect to z or the supplied weights. Sampling also does not unlock exact REINFORCE updates: those require derivatives of the sampling log probability with respect to the trainable parameters. Surrogate gradients or black-box optimization remain necessary here. [4](https://arxiv.org/abs/1308.3432) [12](https://arxiv.org/abs/1506.05254)

## Learning through scalar Jev activations

The public evaluation API documents answers and usage, but no numerical input-gradient interface. An ordinary automatic differentiation library cannot trace derivatives through that remote call. The training system must supply a backward rule. [2](https://docs.typesafe.ai/api)

### Use a surrogate derivative first

Query Jev over a range of scalar inputs using a fixed prompt and number format. Fit a smooth approximation to the measured response curve. During training, run the forward pass through real Jev and use the approximation's slope during the backward pass. For one connection, the proposed update signal is:

```text
estimated dL/dw_i = (dL/da) * surrogate_slope(z) * input_i
```

The derivative is a biased estimate. A fitted curve can also have poor slopes even when its predicted values look accurate, so test whether the resulting weight steps reduce actual Jev-based loss. Refresh its calibration if training visits a different range of inputs.

A simple sigmoid-shaped or straight-through derivative is another baseline. These approaches relate to established work on gradients through hard or stochastic neurons. The cited work provides a method to try, rather than a guarantee for an API that receives serialized numbers. [4](https://arxiv.org/abs/1308.3432)

### Estimate local slopes with extra evaluations

Because Variant A gives Jev only a scalar, we can query nearby inputs and use a central difference as a local slope proxy:

```text
slope(z) = (Jev(z + eps) - Jev(z - eps)) / (2 * eps)
```

Combine this slope with the ordinary derivatives of the local weighted sum to update every incoming weight. Including the original activation, this needs about three Jev evaluations per neuron. There is no need to perturb each weight separately in this variant.

Choose perturbation scales experimentally. If eps is smaller than the serialization precision, both calls can receive identical text. Larger changes can cross abrupt token or answer boundaries. The result is a slope over that finite interval, not exact differentiation through Jev.

### Measure the activation before training

Sweep negative, zero, and positive values, with dense samples near zero. Repeat selected inputs, compare a few fixed precision levels, and measure monotonicity, saturation, and changes under small perturbations. Select the representation on a development set and freeze it for the main comparison.

A nearly perfect sign evaluator can produce saturated outputs and an almost flat local derivative. That would make direct slope estimation unhelpful even though the neuron is answering correctly. A surrogate backward rule may still permit learning, as in networks with hard activations. Binary-network research is relevant context for this distinction. [8](https://arxiv.org/abs/1602.02830)

## Optimizing complete neuron states

Variant B changes the function being optimized. Jev now receives incoming activations, weights, and bias as text. Its output may depend on both the intended arithmetic and how those numbers are represented. We cannot assume it behaves exactly like a weighted sum followed by a threshold.

### A surrogate must model the complete neuron

Fit an approximation from sampled full states to Jev outputs. To backpropagate through multiple layers, the approximation needs useful derivatives with respect to incoming activations as well as each weight and the bias. Weight sensitivities alone cannot pass error to earlier layers.

Keep real Jev evaluations in training forward passes and evaluate loss with those outputs. Use the surrogate only for the backward rule. Validate its update directions against additional real calls on representative states. A network trained entirely on cached or surrogate outputs should be recorded as a different experiment.

### SPSA provides an alternative without local derivatives

Simultaneous perturbation stochastic approximation, or SPSA, estimates an update by changing all trainable parameters together and comparing two losses. It avoids building a differentiable model of each neuron. [5](https://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf)

```text
delta = random vector with entries -1 or +1
loss_plus  = loss(real_Jev_network(theta + c * delta), batch)
loss_minus = loss(real_Jev_network(theta - c * delta), batch)
gradient_i = (loss_plus - loss_minus) / (2 * c * delta_i)
theta = theta - learning_rate * gradient
```

Both evaluations must use the same minibatch. The two losses require two complete network executions, not two API requests. Additional directions or repeated measurements may be needed for useful estimates. Two evaluations per estimate do not imply quick convergence, especially for a discontinuous serialized objective.

Compare SPSA with the surrogate approach on a tiny task before trying a large image dataset. Use a shared evaluation budget so an optimizer cannot appear better simply because it consumed many more neuron evaluations.

### Learning instructions is a later experiment

A further extension could optimize instructions or criteria, changing what each neuron is asked to detect. TextGrad uses textual feedback to improve components of compound AI systems; it is relevant to this extension. Textual feedback is not a numerical derivative, and Jev's typed answers alone would not supply the free-form feedback used by that method. [9](https://arxiv.org/abs/2406.07496)

TypeSafe's feature-discovery cookbook also offers a useful comparison: a generative model proposes questions, TypeSafe turns answers into features, and a supervised model learns from them. That is evidence for combining typed judgments with learning, but it does not demonstrate training a multilayer network of Jev neurons. [10](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery)

## Experiments and evidence

The first milestone is a repeatable learning signal. Scale the dataset only after a small network improves its actual Jev-based loss and classifies unseen examples. Keep the model version, prompts, architecture, preprocessing, and evaluation rules fixed within each comparison.

### A staged experiment

1. Measure one neuron. Sweep scalar inputs, repeat selected calls, and test precision, encoding, and batching. Compare against exact arithmetic. Establish the response curve before attempting training.

2. Train one neuron on AND or OR. Optimize its weights and bias from random starts. Check the complete truth table and loss across several seeds. Compare scalar, precomputed-contribution, and complete-state variants on this small task.

3. Train a hidden layer on XOR. This tests whether learning works through multiple neurons. Compare surrogate gradients, slope probes, and budget-matched SPSA. Record failures as well as successful fits.

4. Learn a nonlinear two-dimensional boundary. Generate labeled points inside and outside a circle. Plot the learned boundary on held-out points so holes, unstable regions, and improvements are visible.

5. Classify two digits. Use native 28 by 28 images of two MNIST classes, such as 0 and 1, with scalar neurons. Compare soft, hard, and sampled activations. Check held-out performance before expanding the number of outputs.

6. Classify all ten digits. Begin with 1,000 training and 1,000 validation examples from MNIST's training partition, stratified by digit. Freeze sampling and preprocessing. Reserve the official 10,000-image test set for final evaluation. MNIST supplies 60,000 training images at 28 by 28 resolution. [6](https://yann.lecun.org/exdb/mnist/index.html)

7. Scale only after validation improves. Compare scalar and full-state neurons on identical splits, topology, and resolution. Reduced-resolution full-state pilots need matched scalar controls. Change data volume or width one at a time and measure the accuracy gained per additional evaluation.

### Comparisons needed to interpret the result

| **Run** | **Forward computation** | **What it tests** |
| --- | --- | --- |
| Conventional baseline | Local activation | Whether the small architecture can solve the task |
| Swap after training | Jev with baseline weights | Damage caused by replacing the activation |
| Train with scalar Jev | Real scalar Jev calls | Whether training adapts the connections |
| Train with full states | Real full-state Jev calls | Effect of moving arithmetic into Jev |

Match preprocessing, topology, initialization seeds where applicable, and training objectives. Include a local hard-activation control with a similar surrogate rule. Tune on validation data and evaluate chosen checkpoints on the untouched test set.

### What to record

Record training and validation loss, held-out accuracy, per-digit errors, activation saturation, and variability across seeds. Track neuron evaluations, HTTP requests, tokens, retries, elapsed time, and measured spend separately. Record the resolved model version and exact prompts with each checkpoint.

An improvement over the swap-after-training run would support adaptation to Jev. A failure to beat chance on held-out digits, persistent flat outputs, or updates that do not reduce real loss would call for revisiting the neuron design before expanding the dataset. No accuracy target is asserted as an expected result.

## Fallbacks when Jev is unreliable

Use several techniques because different failures need different fixes. First measure repeated identical calls, compare intended arithmetic with returned probabilities, and probe small changes to the state. A repeatable but incorrect answer, a saturated activation, and a noisy answer are separate problems.

| **Observed problem** | **Technique to compare** | **Cost and limitation** |
| --- | --- | --- |
| Probability is almost always 0 or 1 | Fitted smooth backward rule; clipped straight-through rule; local hard-activation control | About one live forward pass per update, plus calibration. The backward direction is biased and may fail. |
| Useful local changes exist but the fitted slope is poor | Compare probe sizes in calibration; use one central-difference size per update | About three scalar evaluations per neuron. Extra probe sizes cost more; small probes may vanish during serialization. |
| Full-state behavior is jagged or hard to approximate | SPSA over outer weights and biases | Two network passes per direction. Several directions or many more updates may be needed. |
| Repeated identical calls produce different values | Average a few measured repeats; compare paired losses on the same minibatch | k repeats cost about k times as much. Correlated or systematic errors can remain. |
| Arithmetic errors grow with vector length or cancellation | Shorter vectors; paired input-weight records; precomputed contributions; scalar states | Changes token use and the work assigned to Jev. Compare each representation as its own experiment. |
| Updates have no visible effect or improve only training loss | Check transmitted precision; test larger probes; inspect held-out loss | More calls are useful only if they resolve the failure. Stop scaling until unseen examples improve. |

### A practical comparison order

Try a fitted surrogate and a straight-through rule first on AND, OR, and XOR. Add finite differences only if the response sweep shows meaningful local variation. Test SPSA on the same tiny tasks when local derivatives are unhelpful, then advance the best candidates to digit classification. A tiny quantized network can also use coordinate search as a diagnostic, but searching 25,450 parameters individually is expensive.

For SPSA, keep the plus and minus losses on the same images. If hidden activations are sampled locally, reuse the same random numbers for the paired passes to reduce avoidable comparison noise. This does not control remote model variation. Compare optimizers at equal evaluation or dollar budgets, and record their different numbers of updates.

### Measure whether repetition helps

Do not repeat every neuron by default. Start with three to five repeats of selected states. If outputs are identical, averaging them adds cost without changing the answer. Entropy near p = 0.5 measures ambiguity in the returned Bernoulli distribution; it does not establish run-to-run noise. Different encodings can form an ensemble, but first verify on validation data that their errors differ enough to justify the extra evaluations.

Keep real Jev in the forward path for the main experiment. A cached or surrogate-only network is a useful control, but it answers a different question. Advance only when actual held-out Jev loss and accuracy improve across multiple seeds; a lower surrogate loss alone is insufficient.

## Evaluation budgets and the demo

For the 784 to 32 to 10 candidate, the parameter count is (784 + 1) × 32 + (32 + 1) × 10 = 25,450. A forward pass evaluates 32 + 10 = 42 neurons. These counts assume ten Jev output units as well as the hidden units.

| **Workload** | **Jev neuron evaluations** |
| --- | --- |
| One image forward pass | 42 |
| 1,000 examples for 10 epochs with surrogate backward passes | 420,000 |
| One forward sweep of 60,000 examples | 2,520,000 |
| 60,000 examples for 10 epochs with surrogate backward passes | 25,200,000 |
| One scalar slope based update over B examples including forward values | About 126 × B |
| One SPSA gradient estimate over B examples | 84 × B |

These are arithmetic planning counts, not measured runs. They exclude validation, final testing, calibration, retries, and extra gradient estimates. The 60,000-example cases show full-partition scale; a retained validation split reduces the actual training count. One final evaluation of all 10,000 MNIST test images adds 420,000 neuron evaluations.

### Batching and latency

Independent questions within a layer can be grouped. Later layers still depend on the preceding activations, so this candidate requires two sequential evaluation stages per image. Actual HTTP request counts depend on batching limits and chosen state layout. Batching reduces request overhead and can amortize shared-state tokens; it does not remove the underlying neuron evaluations. [3](https://docs.typesafe.ai/primitives) [14](https://docs.typesafe.ai/cookbooks/parallel_questions)

Compare isolated and batched execution before assuming their behavior matches. Use explicit state references, prevent examples from seeing labels, and avoid sending irrelevant state. Pin the model version and number format. Keep retries distinguishable from new model evaluations in the measurements.

### Estimate cost from a pilot

Measure tokens, billed cost, and end-to-end latency for representative scalar and full-state calls. Use those measurements to estimate each training run, including validation and calibration. The NAND demo's per-gate cost is not a reliable estimate for longer vector states. A daily or per-run spending cap should be chosen before live training.

### The visible result

The demo should let someone draw a digit and follow the actual activations through the network. Show learned first-layer weight maps, ten output activations, a training curve, and the running inference bill. Label a recorded training replay as a replay; keep live inference tied to real responses.

The strongest comparison is a digit that fails after Jev is substituted into a conventional network but succeeds after training with Jev. Show aggregate held-out results beside the example so one favorable prediction does not carry the claim. The research result would be evidence that learned connections can adapt to a frozen model's imperfect neuron behavior.

## Rough API cost estimates

Jev 1.13.0 is listed at $0.042 per million input tokens; outputs are free. The estimates below use that price as checked on September 26, 2026. No paid experiment has been run for this proposal. [13](https://docs.typesafe.ai/models)

```text
API cost in USD = neuron evaluations * T * 0.042 / 1,000,000
T = total billed input tokens / total neuron evaluations
```

T includes state, questions, and overhead after batching. The token columns are assumptions, not measured averages or bounds; verbose vectors can exceed 1,500. Measure scalar and complete-state designs separately. Estimates exclude local compute and any separate model used to revise prompts.

| **Workload** | **Evaluations** | **T = 100** | **T = 400** | **T = 1,500** |
| --- | --- | --- | --- | --- |
| Calibration allowance | 100,000 | $0.42 | $1.68 | $6.30 |
| 1,000 images × 10 epochs | 420,000 | $1.76 | $7.06 | $26.46 |
| Final 10,000-image test | 420,000 | $1.76 | $7.06 | $26.46 |
| 60,000 images × 1 epoch | 2,520,000 | $10.58 | $42.34 | $158.76 |
| 60,000 images × 10 epochs | 25,200,000 | $105.84 | $423.36 | $1,587.60 |
| Nine-run pilot described below | 11,860,000 | $49.81 | $199.25 | $747.18 |

Training rows assume real forward passes with local surrogate backward passes. Validation, retries, and calibration are excluded unless specified. Holding out validation images reduces the full-partition training counts.

### Training method changes the bill

At equal minibatches and update counts, scalar central differences cost about 3 times the training evaluations. SPSA costs about 2K times for K directions per update. Three seeds multiply training and validation by three; five repeats multiply affected costs by five. These factors do not imply equal convergence. Extra loss checks add evaluations.

### A concrete small comparison study

Three methods at training multipliers 1, 2, and 3, each with three seeds and ten passes over 1,000 images, require 7.56 million training evaluations. Checking 1,000 validation images after each pass adds 3.78 million. Add 100,000 calibration evaluations and one final 10,000-image test for the selected checkpoint: 11.86 million total, about $199 at T = 400. This example holds update counts fixed; comparisons at equal cost should give each method the same evaluation allowance.

### Start with a spending cap

Start with a $25 feasibility cap and a small token-accounting probe. If learning is repeatable, a proposed $250 cap covers the middle-column pilot with some reserve; at T = 1,500 the same study is about $747 before retries. Reduce scope or stop at the cap. Neither amount guarantees a successful classifier.

The supplied NAND bill implies about $0.0000155 per gate. At that cost per neuron, the small training row would be $6.52 and the full ten-epoch row $391. State size and batching can change this. Log actual usage and elapsed time; low token cost does not establish fast training.

## Sources and their relevance

Primary sources accessed on 26 September 2026. API descriptions establish the available interface; the papers establish related optimization methods. The proposed Jev classifier remains untested.

**1. TypeSafe AI.** [Noul](https://docs.typesafe.ai/primitives/noul)

Defines the yes probability used as a neuron activation.

**2. TypeSafe AI.** [API reference](https://docs.typesafe.ai/api)

Documents state, typed questions, answers, usage, and model identifiers. No input-gradient interface is documented.

**3. TypeSafe AI.** [Primitives](https://docs.typesafe.ai/primitives)

Explains independent questions, shared state, and batching. Sequential layers still need earlier outputs.

**4. Bengio, Léonard, and Courville, 2013.** [Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation](https://arxiv.org/abs/1308.3432)

Relevant foundation for surrogate and straight-through gradient estimators.

**5. Spall, 1992.** [Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation](https://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf)

Primary paper for SPSA, which estimates an update using two perturbed objective evaluations.

**6. LeCun, Cortes, and Burges.** [The MNIST Database](https://yann.lecun.org/exdb/mnist/index.html)

Dataset reference for 60,000 training images and 10,000 test images, originally 28 by 28 pixels.

**7. TypeSafe AI.** [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

Version-specific limitations on numeric precision and arithmetic; reviewed September 17, 2026.

**8. Courbariaux et al., 2016.** [Binarized Neural Networks](https://arxiv.org/abs/1602.02830)

Related work on training networks with binary weights and activations; not evidence about Jev.

**9. Yuksekgonul et al., 2024.** [TextGrad Automatic Differentiation via Text](https://arxiv.org/abs/2406.07496)

Uses textual feedback to improve compound AI systems; relevant to later prompt optimization.

**10. TypeSafe AI.** [Autoresearch feature discovery](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery)

A related workflow learns from TypeSafe-derived features and revises questions using model errors.

**11. TypeSafe AI.** [State](https://docs.typesafe.ai/concepts/state)

Defines request state and supported text and JSON representations. State is context supplied to a question.

**12. Schulman, Heess, Weber, and Abbeel, 2015.** [Gradient Estimation Using Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254)

Related theory for stochastic computation and gradient estimators. Sampling a Jev output does not expose its parameter derivatives.

**13. TypeSafe AI.** [Models](https://docs.typesafe.ai/models)

Current Jev 1.13.0 pricing is $0.042 per million input tokens, with free outputs. Pricing checked September 26, 2026.

**14. TypeSafe AI.** [Parallel questions](https://docs.typesafe.ai/cookbooks/parallel_questions)

Demonstrates billing shared state once per request and measuring tokens and repeated-call variability. Its savings are workload-specific.

**15. TensorFlow.** [Quickstart for beginners](https://www.tensorflow.org/tutorials/quickstart/beginner)

Conventional MNIST reference using 784 inputs, 128 hidden units, and 10 outputs. Its reported accuracy does not establish Jev performance.

**16. Stanford CS231n.** [Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)

Explains weighted sums, activations, learned parameters, backpropagation, and architecture choices.
