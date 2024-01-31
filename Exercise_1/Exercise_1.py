# Data.Ml.100 Introduction to Pattern Recognition and Machine Learning
# Exercise 1

# Importing Necessary Libraries
import sys

# Read command line arguments and convert to a list of integers
arr = sys.argv[1].split(',')

# Creating a list
my_numbers = [None]*len(arr)

for idx, arr_val in enumerate(arr):
    my_numbers[idx] = int(arr_val)

# Print
print(f'Before sorting {my_numbers}')

# My sorting (e.g. bubble sort)
# ADD HERE YOUR CODE
def bubbleSort(array):
    
  # loop to access each array element
  for i in range(len(array)):

    # loop to compare array elements
    for j in range(0, len(array) - i - 1):

      # compare two adjacent elements
      # change > to < to sort in descending order
      if array[j] > array[j + 1]:

        # swapping elements if elements
        # are not in the intended order
        temp = array[j]
        array[j] = array[j+1]
        array[j+1] = temp


bubbleSort(my_numbers)

# Print
print(f'After sorting it is {my_numbers}')
