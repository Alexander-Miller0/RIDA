# Random Inverse Depolarizing Approximation (RIDA)

Universally applicable, low overhead quantum error mitigation for gate and measurement error. We present the theory and results behind RIDA in our paper the accompanying paper: A. X. Miller, M. B. Soley, "Universal Quantum Error Mitigation via Random Inverse Depolarizing Approximation," in preperation.
 
 # Instructions
 
 To create the depolarizing estimation circuit, run the following:
 
 ```
random_inverse(your_circuit)
 ```
 Then run multiple such estimation circuits and average the resultant expectation values. The expectation values obtained from target circuits can be divided by this average expectation value to provide an error-mitigated expectation value.

 Additional documentation is included in documentation.pdf.
