# NLP Prompt Optimizer - Test Prompts

This document contains example prompts that have been tested with the NLP Prompt Optimizer, demonstrating the various transformation capabilities of the system.

---

## 1. Energy/Sustainability Phrase Removal

| Original Prompt | Optimized Result |
|----------------|------------------|
| "Using renewable energy sources, explain machine learning" | "Explain machine learning" |
| "In a sustainable manner, describe neural networks" | "Describe neural networks" |
| "With minimal carbon footprint, analyze this data" | "Analyze this data" |
| "Leveraging green computing practices, summarize the article" | "Summarize the article" |

---

## 2. Verbose Phrase Transformations

| Original Prompt | Optimized Result |
|----------------|------------------|
| "I would like you to explain the concept of AI" | "Explain AI" |
| "Could you please help me understand deep learning?" | "Explain deep learning" |
| "I was wondering if you could describe clustering" | "Describe clustering" |
| "Would it be possible for you to summarize this text?" | "Summarize this text" |
| "I need you to analyze the following data" | "Analyze the following data" |

---

## 3. Filler Word Removal

| Original Prompt | Optimized Result |
|----------------|------------------|
| "Basically, explain how neural networks work" | "Explain how neural networks work" |
| "Actually, I need help with classification" | "I need help with classification" |
| "So, essentially, describe the algorithm" | "Describe the algorithm" |
| "Just explain the concept of regression" | "Explain the concept of regression" |

---

## 4. Passive to Active Voice

| Original Prompt | Optimized Result |
|----------------|------------------|
| "The data should be analyzed by you" | "Analyze the data" |
| "The model needs to be trained" | "Train the model" |
| "The results should be summarized" | "Summarize the results" |
| "The text needs to be processed" | "Process the text" |

---

## 5. Word Simplifications (JSON-based)

| Original Prompt | Optimized Result |
|----------------|------------------|
| "Utilize this algorithm to accomplish the task" | "Use this algorithm to do the task" |
| "Commence the analysis and terminate when complete" | "Start the analysis and end when complete" |
| "Demonstrate how to implement the solution" | "Show how to implement the solution" |
| "Ascertain the optimal parameters" | "Find the best parameters" |

---

## 6. Redundant Phrase Removal

| Original Prompt | Optimized Result |
|----------------|------------------|
| "In order to understand the basics fundamentals" | "To understand the fundamentals" |
| "Due to the fact that the model performs well" | "Because the model performs well" |
| "At this point in time, analyze the data" | "Now, analyze the data" |
| "For the purpose of training the model" | "To train the model" |

---

## 7. Duplicate Content Detection

| Original Prompt | Optimized Result |
|----------------|------------------|
| "Explain ML. Explain ML. Explain ML." | "Explain ML." |
| "Analyze data. Analyze data." | "Analyze data." |
| "Train the model. Train the model. Train the model." | "Train the model." |

---

## 8. Complex Multi-Transformation Examples

### Example 1: Multiple Transformations
**Original:**
```
I would like you to, in a sustainable manner, basically explain how 
neural networks work and the results should be summarized by you.
```

**Optimized:**
```
Explain how neural networks work and summarize the results.
```

**Transformations Applied:**
- Removed "I would like you to" (verbose)
- Removed "in a sustainable manner" (energy phrase)
- Removed "basically" (filler)
- Converted "should be summarized by you" to active voice

### Example 2: Verbose with Fillers
**Original:**
```
Could you please help me understand, essentially, the concept of 
machine learning and, actually, how it can be utilized?
```

**Optimized:**
```
Understand the concept of machine learning and how it can be used?
```

### Example 3: Technical with Redundancy
**Original:**
```
In order to accomplish the task of training the model, utilize the 
following methodology and demonstrate the results.
```

**Optimized:**
```
To do the task of training the model, use the following methodology 
and show the results.
```

---

## 9. Edge Cases

| Scenario | Original | Result |
|----------|----------|--------|
| Empty input | "" | "" |
| Already optimal | "Explain AI" | "Explain AI" |
| Very short | "Help" | "Help" |
| Only fillers | "Basically just" | "" |
| Special characters | "Explain AI! @#$" | "Explain AI! @#$" |

---

## 10. Real-World Use Cases

### Academic Writing
**Original:**
```
I was wondering if you could help me understand, in a sustainable 
and environmentally conscious manner, the fundamental basics of 
how deep learning neural networks are utilized in order to 
accomplish the task of image classification.
```

**Optimized:**
```
Explain how deep learning neural networks are used for image classification.
```

### Data Science Query
**Original:**
```
Could you please, using renewable energy principles, analyze the 
following dataset and the results should be visualized by you in 
a comprehensive manner.
```

**Optimized:**
```
Analyze the following dataset and visualize the results comprehensively.
```

### Code Assistance
**Original:**
```
I would like you to basically help me implement, leveraging green 
computing practices, a function that essentially performs sorting 
on the array.
```

**Optimized:**
```
Implement a function that performs sorting on the array.
```

---

## Token Reduction Statistics

| Prompt Type | Avg. Original Tokens | Avg. Optimized Tokens | Reduction |
|-------------|---------------------|----------------------|-----------|
| Simple verbose | 12 | 4 | 67% |
| With fillers | 15 | 8 | 47% |
| Energy phrases | 10 | 5 | 50% |
| Complex multi | 35 | 12 | 66% |
| Real-world | 40 | 15 | 62% |

---

## How to Add New Test Prompts

1. Add prompts to appropriate category above
2. Run through the optimizer to verify results
3. Document any edge cases discovered
4. Update statistics if significant

---

*Last Updated: December 2024*
*Maintained by: Sustainable AI G3 Team*
