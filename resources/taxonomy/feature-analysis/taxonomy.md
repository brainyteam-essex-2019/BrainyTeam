# Feature Analysis

## ERP Analysis

### General

- [10.1093/schbul/sbn093](https://dx.doi.org/10.1093%2Fschbul%2Fsbn093)
    - Elides noise by aggregating over large quantity of epochs,
    so that the peaks amplitudes and the latencies are valuable for identifying
    discrete sensory and cognitive processes.
    - > "Assumes that ERP components reflect transient bursts of neuronal activity,
      > time locked to the eliciting event, that arise from one or more neural generators
      > subserving specific sensory and cognitive operations during information processing. "
      - Depending on the assumption that "noise" is completely independent
      of the processing of the task events
      - But actually the background oscillation has played
      their role in the task processing
    - Phase resetting may generate "fake actives" in ERP analysis

## Time-frequency Analysis

### General

- [10.1093/schbul/sbn093](https://dx.doi.org/10.1093%2Fschbul%2Fsbn093)
    - Time-frequency analysis has been used for schizophrenia studies
    - Comparing to ERP, it has potential to analysis parrallel processing
    of information inside a brain
    - "provide additional information about neural synchrony not apparent in the ongoing EEG"


#### Neuronal Oscillation

- [10.1093/schbul/sbn093](https://dx.doi.org/10.1093%2Fschbul%2Fsbn093)
    - Neural oscillation and neural synchrony has represented important
    mechanisms for cognition process.
    - Oscillation analysis relies on spectral decomposition

- [10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C](https://doi.org/10.1002/(SICI)1097-0193(1999)8:4%3C194::AID-HBM4%3E3.0.CO;2-C)
    - Cognitive acts depends on integration of large amounts of functional
    areas accross the brain
    - Phase synchrony should subserve all dimensions of a cognitive act.

#### Coherence Analysis
- [10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C](https://doi.org/10.1002/(SICI)1097-0193(1999)8:4%3C194::AID-HBM4%3E3.0.CO;2-C)
    - It does not separate effects of amplitude and phase in the interrelations between two signals
    - It works **only** for stationary signals
    - Does not specifically quantify phase relationships

#### Phase Locking

- [10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C](https://doi.org/10.1002/(SICI)1097-0193(1999)8:4%3C194::AID-HBM4%3E3.0.CO;2-C)
    - > "To examine the role of neural synchronies as a putative
        > machanism for long-range neural integration during cognitive tasks"
    - > Measures the significance of the phase covariance between two signals
        > with a reasonable time-resolutio
    - _Phase locking value_ calculates for each latency a measure
    of phase-locking between different components given a frequency
        - 1 means two components are strongly phase-locked,
        0 means the opposite
    - Cons:
        - Depends crucially on the choice of the frequency $f$
        - Assumes that the likelihood of the phase-difference
        between two components stays the same across trials