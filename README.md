# Scientific ML/AI

This is an open study group to level-set on various approaches to leveraging
machine learning tools and techniques when solving scientific problems.



## What do you mean by Study Group?

When Quantum Computing first became a thing we gathered a multidisciplinary
group at UT Austin... folks from the CS, Math, and Physics departments met
weekly and we'd teach the CS folks quantum mechanics and learn about
algorithmic complexity and coding theory from them in return.  Following the
natural order of things, the math crew would heckle from the sidelines trying
to clean up details on both sides.

I'd like to reproduce something similar here.  We'll keep it to a reading group
at first, but I expect we'll work this into open-source course material going
forward.

There's an outline for a core curriculum below as well as an arbitrarily
ordered reading list.  The idea is we'll take turns co-presenting from the
reading list as we fill gaps in the core material on an individual basis.
Pretty much everything in the outline below would be considered "background"
for someone in the group and brand new for another... that's expected.

---

# Target audience

Potential learners:

- Researchers: know the science, new to deep learning
- Data Scientists: know the ML, new to the science
- Undergrads: new to both

---

# Some questions to answer

- What is SciML/AI?

- How can I use deep models to speed up my montecarlo runs?

- Can I use physics to cut down the training time and/or data for my AI models?

- Can I use deep learning to improve the efficitency and time execution for
  batch processing workloads?

- Can I use a GPU/TPU to run my {thermal, signal prop, weather, etc} sims?

- What's the best way to run deep models (or general SciML models)?

- How do I actually schedule hybrid models across hetero resources?

- What do the data pipelines look like for SciML/AI workloads?

- Do I have to rewrite all my code in Julia or Swift?  Where does JAX, XLA, etc
  fit in?  How can I leverage different accelerators?

- Are there off-the-shelf models I can share or use?  E.g., VNN19.  AI Hub for
  custom/trained models?

- Who's actually doing any of this?  CERN, Fermilab, etc?

- Where can I find good examples?

- Does it actually work?

- What are the limitations?

- Identify apps that don't run well on GPUs (like genomics workloads)... are
  there other apps that do _not_ work well with these techniques?

---

# Some topics to cover

The content can be broken down into *Core* -vs- *Background* material.  We
don't want to rabbit-hole too much on background, but it's useful to include it
so folks can dig further if they want.

We can consider this the body of material to learn.  Each paper or example we
walk through fills in some of these blanks.

## Calc

(Background)

- autodiff
- src-to-src -vs- runtime
- differentiable programming
- what do languages like julia and swift bring to the picture?
- motivation from SGD etc in ML
- we're implicitly assuming enough familiarity with linear algebra


## DiffEq

(Background)

- ODEs:
  - the physics
  - dealing with ICs
- PDEs:
  - the physics
  - BCs
  - Green's functions?
- nonlinear PDEs:
  - the physics
  - limits of analytical methods
- numerical methods:
  - FE basics
  - WRF, OpenFOAM, thermal, signal prop, etc
  - ? anything other than MPI at this point ?
- stochastic methods:
  - montecarlo
  - use example from CERN or maybe FinTech will be more familiar?


## NeuralNets

(Background)

- basic intro
- examples
  - style transfer
  - plane-or-not


## Neural DiffEqs

(Core)

- resnets to ODEs
- maybe applications to time-series data?  maybe applications to
  non-time-series data?

## Modeling and Simulation

(Background)

- general approach
- continuous -vs- discreet time/space
- analytical, numerical, stochastic/non-determ, etc
- simulation process / frameworks / IDEs
- examples in use

## Mixed models

(Core)

- DL-fed traditional models
  - does MC fit here?
- Black box DL models
- various hybrids
  - surrogate modeling
  - adjoint sensitivity analysis
  - inverse problems
  - probablilistic programming models
- speed up training or train on sparse data via physics-informed modeling


## Model management

(Core)

- M&S
- Verification, Validation, and Accredation (VV&A)
- mixed model management
- simulation management
- training is special?
- generative approaches?
- adversarial approaches?
- mesoscale modeling?
- maybe there are some techniques we can steal from RL?
  e.g., trebuchet as RL env?


## Infrastructure for all of this

(Core)

- what's the infrastructure look like for this?
- available compute resources... how do I decide?
- simulation management requirements
- pipelines for HITL, MITL, etc?
- data management... we could have a whole topic on this
- security, logging/auditing, versioning, etc etc... all the devopsy things
- automation


## ML Ops

(Core)

- TFX
- kubeflow
- data issues (io/staging/access/etc)
- security, logging/auditing, versioning, etc etc... all the devopsy things
- automation
- hybrid HPC/ML OPs (should this be a separate topic?  like "SciML Ops")
  - scheduling across heterogeneous resources

---

# Comments on gaps so far

## Jamie

Focus on SciML _infrastructure_!

- optimize utilisation of clusters
- scale of work
- VM shape/type
- region or another
- degree of parallelism of individual jobs
- ML can help dictate infra
- quick returns
- cheap runs
- quicker development/training
- what's the optimal construction of ensembles of models?


---

## Straight from Chris Rackaukas

The types of tools which are necessary for large-scale scientific ML are:

- Tooling for solving neural differential equations
- Differentiable programming (automatic differentiation) tools
- Probabilistic programming tools to learn uncertainty from data
- Helper tools for sparsity detection and sparse differentiation
- Structured linear algebra tools
- Number types for mixed precision arithmetic
- Methods for discretizing partial differential equations
- Tools for generating and utilizing GPU kernels
- Uncertainty quantification and Global sensitivity analysis
- Surrogate modeling techniques


### Example

The Defense Advanced Research Projects Agency (DARPA) Defense Sciences Office
(DSO) is requesting information on state-of-the-art approaches to generate
multi-physics modeling and simulation codes directly from a description of the
physical phenomena. Of interest are modeling and simulating increasingly
complex systems involving multiple physics that require high fidelity
simulations but have limited test data (e.g., combustion, hypersonics, nuclear
stockpile).

One way to approach this with scientific ML would be to do the following:

- Build an Natural Language Processing (NLP) stack that interprets text into
  PDEs
- Autodiscretize and solve the PDE
- Write a loss function which checks the PDE solution against data
- Add regularization based on the global sensitivity and uncertainty of the
  solution


---

## Evaluating papers

From [paco3637](https://www.cs.colorado.edu/~paco3637/),
here's a list of possible dimensions we can use to evaluate papers:

- What is the science or engineering problem(s) the paper is addressing?
- What is the learning method they use? Provide some mathematical background
  and general interpretation.
- Is the method appropriate for the problem? Why or why not?
- Does the learning method perform better than existing approaches?
- Does the learning method solve a problem that could not previously be solved?
- Does the paper claim any significant results beyond outlining opportunities
  and challenges in future work?
- Supposing the paper uses established methods with existing and accepted
  terminology, why use the AI/ML/Data buzzwords? What does this buy (other than
  hype)?
- Do you think the paper's approach should be widely adopted? Why or why not?

He also includes this list of
[paywalled papers](https://www.cs.colorado.edu/~paco3637/sciml-refs.html)
from Spring2019.
