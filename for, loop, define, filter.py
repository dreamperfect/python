# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:52:53 2024

@author: 16692
"""

#have to select the entire loop before running it



# % sign is "modulo operator" which returns the remainder after division 

#so a%b outputs the remainder when a is divided by b. Note that this is different than returning 
# the decimals after a division. For example 10/3=3.33333. But if you divide 10 by 3 you get
# 3 remainder 1, that 1 then gets "turned into 10" by adding an imaginary 0 to then divide 3 again to get you 3.3,
# you then get another remainder of 1 which gets turned into 10 again to repeat the process until you keep getting 3.33333


10%3 #outputs 1

#note that if the right number is bigger it will not divide the number and just return 
#the first one as a "remainder"

2.64 % 3





##18/7 = 2 remainder 4 which in division gets turned into 40 so then its 40/7= 5 remainder 5,
# which gets turned to 50 so its then 50/7= 7 remainder 1 which gets turned to 10 and division continues...

18/7   #output 2.5714285714285716

18%7 #outputs just the first remainder = 4
 




print(list(range(10)))

#loop that outputs the numbers from 0 to 9 that are "not divisible by 2" (as whole numbers)
for i in range(10):
    if ((i%2)!=0):
        print(i)

#outputs 1 3 5 7 9 




#create a range object that reppresents a sequence of numbers starting at 1, incrementing by 1 to 9999

#range(start (inclusive),stop (exclusive),step (default 1 if left blank))

nums=range(1,1000)


print(nums)

print(list(nums))


#prime numbers can only be divided evenly by 1 and itself 
#create a loop that returns true or false statements that you can use to filter

def is_prime(number):
    for x in range(2,number):
        if (number%x)==0:
            return False
    return True


is_prime(7)

is_prime(8)


#filter takes a function (here its is_prime), and an iterable (here its nums), and applies the function to each
# number in nums 

primes=filter(is_prime,nums)

#just a filter object 
print(primes)


primes_list = list(primes)

print(primes_list)

#note that iterators are consumed each time you run list(primes), but you can do


primes=filter(is_prime,nums)

print(list(primes))
