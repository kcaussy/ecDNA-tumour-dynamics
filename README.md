# ecDNA Tumour Dynamics
A stochastic, agent-based model of **extrachromosomal DNA (ecDNA) dynamics** 
during tumour evolution and treatment response.

## Overview

Approximately 17% of cancers harbour amplified oncogenes on extrachromosomal DNA (ecDNA), with
ecDNA prevalence increasing three-fold in more aggressive tumours. These segments of amplified DNA
undergo random segregation, allowing rapid and heritable changes in ecDNA copy number which contributes to
tumour evolution and drug resistance. 

This model simulates ecDNA dynamics that drive tumour evolution and treatment resistance:
- **Reintegration**: ecDNA integrates into a chromosome, forming a homogeneously staining region (HSR) - a second amplification state.
- **Excision**: the reverse - an integrated ecDNA is excised back from the chromosome, returning the HSR
  amplification to an ecDNA amplification.
  
## Project aims

1. Investigate how reintegration and excision shape copy-number distributions and population composition
   during tumour evolution in the absence of therapy.
2. Implement a three-phase treatment protocol to investigate its influence on tumour dynamics, including changes
   in population composition and copy-number distributions, and to determine whether ecDNA dynamics provide a
   mechanistic explanation for therapeutic resistance.
