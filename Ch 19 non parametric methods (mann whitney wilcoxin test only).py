# -*- coding: utf-8 -*-
"""


"two-sided" → Two-tailed test

𝐻o: The two distributions are identical.
Ha:The distributions differ in location (not specifying direction).
	​
"less" → One-tailed (lower)

𝐻0: The distribution of sample 1 ≥ sample 2.
Ha:The distribution of sample 1 < sample 2.

"greater" → One-tailed (upper)
𝐻o:The distribution of sample 1 ≤ sample 2.
Ha:The distribution of sample 1 > sample 2.



Reject the null hypothesis (H₀) if 𝑝-value ≤ 𝛼


Do not reject H₀ if 𝑝-value >𝛼

"""


from scipy.stats import mannwhitneyu

batch_1 = [15,3,23,8]
batch_2 = [18,20,32,9,25]


#by default if you dont specify alternative= it goes to two tailed test
stat, p_value = mannwhitneyu(batch_1, batch_2)

print('Statistics=%.2f, p=%.4f' % (stat, p_value))
alpha = 0.05
if p_value < alpha:
    print('Reject Null Hypothesis (Significant difference between two samples)')
else:
    print('Do not Reject Null Hypothesis (No significant difference between two samples)')
    
    

# Two-tailed
stat, p_value = mannwhitneyu(batch_1, batch_2, alternative='two-sided')

print('Statistics=%.2f, p=%.4f' % (stat, p_value))
alpha = 0.05
if p_value < alpha:
    print('Reject Null Hypothesis (Significant difference between two samples)')
else:
    print('Do not Reject Null Hypothesis (No significant difference between two samples)')


# One-tailed (batch_1 < batch_2)
stat, p_value = mannwhitneyu(batch_1, batch_2, alternative='less')

print('Statistics=%.2f, p=%.4f' % (stat, p_value))
alpha = 0.05
if p_value < alpha:
    print('Reject Null Hypothesis, therefore the distribution of sample 1 < sample 2)')
else:
    print('Do not Reject Null Hypothesis, so therefore (The distribution of sample 1 is ≥ sample 2.)')



# One-tailed upper test (batch_1 > batch_2)

stat, p_value = mannwhitneyu(batch_1, batch_2, alternative='greater')

print('Statistics=%.2f, p=%.4f' % (stat, p_value))

if p_value < alpha:
    print('Reject Null Hypothesis, therefore The distribution of sample 1 > sample 2')
else:
    print('Do not Reject Null Hypothesis, so therefore the distribution of sample 1 is ≤ sample 2')


####Other example 


batch_1 = [1095, 955, 1200, 1195, 925, 950, 805, 945, 875, 1055, 1025, 975]
batch_2= [885, 850, 915, 950, 800, 750, 865, 1000, 1050, 935]

#by default if you dont specify alternative= it goes to two tailed test
stat, p_value = mannwhitneyu(batch_1, batch_2)

print('Statistics=%.2f, p=%.4f' % (stat, p_value))
alpha = 0.05
if p_value < alpha:
    print('Reject Null Hypothesis (Significant difference between two samples)')
else:
    print('Do not Reject Null Hypothesis (No significant difference between two samples)')
    