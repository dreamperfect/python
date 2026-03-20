# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:52:53 2024

@author: 16692
"""

#have to select the entire loop before running it



# % sign is module which returns the remainder of the division of the number to
#the left by the number on the right. So 1%2  = the remainder of dividing 2 by 1


18%7


##18/7 = 2 remainder 4 which in division gets turned into 40 so then its 40
#divided by 7 =5.714285 but that is just a short cut to add to the end of 2.57142

18/7



for i in range(10):
    if ((i%2)!=0):
        print(i)



nums=range(1,1000)



print(nums)

print(list(nums))




def is_prime(number):
    for x in range(2,number):
        if (number%x)==0:
            return False
    return True

primes=filter(is_prime,nums)

#just a filter object 
print(primes)


primes_list = list(primes)

primes_list
