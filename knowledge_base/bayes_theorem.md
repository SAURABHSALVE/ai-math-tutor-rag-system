# Bayes' Theorem and Conditional Probability

## Conditional Probability
- P(A|B) = P(A intersect B) / P(B), where P(B) > 0
- Interpretation: probability of A given that B has occurred

## Multiplication Rule
- P(A intersect B) = P(A|B) * P(B) = P(B|A) * P(A)

## Bayes' Theorem
P(A_i|B) = P(B|A_i) * P(A_i) / P(B)

where P(B) = sum_{j} P(B|A_j) * P(A_j) (law of total probability)

### Steps to Apply
1. Identify the "causes" A_1, A_2, ..., A_n (partition of sample space)
2. Find prior probabilities P(A_i)
3. Find likelihoods P(B|A_i)
4. Compute P(B) using total probability
5. Apply Bayes' formula

## Classic JEE Problems

### Urn Problems
- Multiple urns with different ball compositions
- A ball is drawn and has certain property
- Find probability it came from specific urn

### Medical Testing
- Disease prevalence: P(Disease)
- Test sensitivity: P(Positive|Disease)
- Test specificity: P(Negative|No Disease)
- Find P(Disease|Positive) using Bayes

### Manufacturing Defects
- Multiple machines produce items
- Each machine has different defect rate
- Given defective item, find which machine produced it

## Independence
- A and B are independent iff P(A|B) = P(A)
- Equivalently: P(A intersect B) = P(A) * P(B)
- If A, B independent: A, B' also independent
