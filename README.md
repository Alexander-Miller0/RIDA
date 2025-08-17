# Random Inverse Depolarizing Approximation (RIDA)

Universally applicable, low overhead quantum error mitigation for gate and measurement error. 

The theory and results behind RIDA are presented in the accompanying paper: 
Alexander X. Miller, Micheline B. Soley, *Universal Quantum Error Mitigation via the Random Inverse Depolarizing Approximation*, in preparation.
 
 # Instructions
 
 To create the depolarizing estimation circuit, run the following:
 
 ```
random_inverse(your_circuit)
 ```
 Then run multiple such estimation circuits and average the resultant expectation values. The expectation values obtained from target circuits can be divided by this average expectation value to provide an error-mitigated expectation value.

 Additional documentation is included in documentation.pdf.
