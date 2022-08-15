from typing import List
import numpy as np
from heapq import heappush, heappop
from bisect import bisect, insort
from sys import stdin, stdout

def read():
    return stdin.readline().rstrip('\n')

def read_array(sep=None, maxsplit=-1):
    return read().split(sep, maxsplit)

def read_int():
    return int(read())

def read_int_array(sep=None, maxsplit=-1):
    return [int(a) for a in read_array(sep, maxsplit)]

def write(*args, **kwargs):
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    stdout.write(sep.join(str(a) for a in args) + end)

def write_array(array, **kwargs):
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    stdout.write(sep.join(str(a) for a in array) + end)


class Solution:
    def __init__(self):
        ...    
    
    @classmethod
    def roc_auc(self, target: List[float], value: List[float]) -> float :
        '''
        Calculate ROCAUC

        find number inversion in index_sorted_by_value_target
        '''
        target = np.array(target)
        value = np.array(value)

        index_sorted_value = np.argsort(value, kind='stable')
        sorted_target = np.sort(target)
        sorted_value = np.sort(value)
        
        sorted_by_value_target = target[index_sorted_value]
        index_sorted_by_value_target = np.argsort(sorted_by_value_target, kind='stable')

        deb1 = self._number_of_non_inversions(index_sorted_by_value_target)
        deb2 = self._number_of_non_inversions(sorted_target)
        deb3 = self._count_combinations_of_pairs(sorted_target)
        deb4 = self._count_equal_target(sorted_value, sorted_by_value_target)

        numerator = self._number_of_non_inversions(index_sorted_by_value_target) - \
                            self._count_combinations_of_pairs(sorted_target) - \
                            self._count_equal_target(sorted_value, sorted_by_value_target)/4
        
        denominator = self._number_of_non_inversions(sorted_target) - \
                        self._count_combinations_of_pairs(sorted_target)
        
        return numerator / denominator

    @classmethod
    def _count_combinations_of_pairs(cls, sorted_arr: List[float]) -> int:
        res = 0
        right = 0
        left = 0
        while True:
            while (sorted_arr[right] == sorted_arr[left]):
                right += 1
                if right == len(sorted_arr):
                    len_of_equal_elems = right - left
                    res += len_of_equal_elems * (len_of_equal_elems - 1) / 2 
                    return int(res)
            len_of_equal_elems = right - left
            res += len_of_equal_elems * (len_of_equal_elems - 1) / 2
            left = right

    @classmethod
    def _number_of_non_inversions(cls, arr: List[float]) -> int:
        N = len(arr)
        if N <= 1:
            return 0
    
        sortList = []
        result = 0
    
        # heapsort, O(N*log(N))
        for i, v in enumerate(arr):
            heappush(sortList, (v, i))
    
        x = []  # create a sorted list of indexes
        while sortList:  # O(N)
            v, i = heappop(sortList)  # O(log(N))
            # find the current minimum's index
            # the index y can represent how many minimums on the left
            y = bisect(x, i)  # O(log(N))
            # i can represent how many elements on the left
            # i - y can find how many bigger nums on the left
            result += y
    
            insort(x, i)  # O(log(N))
    
        return result

    @classmethod
    def _number_of_non_inversions_in_equal_target(cls, arr: List[float]) -> int:
        try:
            return cls._number_of_non_inversions(arr) - cls._count_combinations_of_pairs(arr)
        except: return 0
  
    @classmethod
    def _count_equal_target(cls, _sorted_value: List[float], _sorted_by_value_target: List[float]) -> int:
        res = 0
        right = 0
        left = 0
        while True:
            while (_sorted_value[right] == _sorted_value[left]):
                right += 1
                if right == len(_sorted_value):
                    res += cls._number_of_non_inversions_in_equal_target(_sorted_by_value_target[left:right]) 
                    return int(res)
            res += cls._number_of_non_inversions_in_equal_target(_sorted_by_value_target[left:right])                    
            left = right
        

sol = Solution()

#print(sol._number_of_non_inversions([0, 2, 1]))
#print(sol._number_of_non_inversions([4, 5, 0, 1, 3, 6, 7, 8, 2, 9]))

print(sol._number_of_non_inversions_in_equal_target([0, 2, 4]))
                                                               
#print(sol._count_equal_target([0, 1, 1, 2, 2, 2, 3, 4, 4, 4],[2, 2, 4, 3, 0, 1, 3, 3, 3, 4]))
#print(sol._count_equal_target([0, 1, 1, 2, 2, 2, 3, 4, 4, 4],[2, 2, 4, 3, 0, 1, 3, 3, 3, 4]))

#print(sol._count_combinations_of_pairs(sorted([0, 0, 1, 0, 1, 1, 1])))
'''
print(sol.roc_auc(target=[0, 1], value=[0, 1])) 
print(sol.roc_auc(target=[0, 1], value=[1, 0]))
print(sol.roc_auc(target=[0.5, 0.5, 2], value=[0, 1, 0.5]))
print(sol.roc_auc(target=[0, 0, 1, 0, 1, 1, 1], value=[0, 1, 2, 3, 4, 5, 6]))
'''
print(sol.roc_auc(target=[0, 3, 1, 2, 1, 2, 4, 2, 4, 0], value=[4, 0, 2, 4, 0, 1, 1, 1, 4, 0]))

print(sol.roc_auc(target=[3, 3, 2, 3, 4, 0, 2, 1, 4, 3], value=[4, 4, 1, 2, 1, 2, 0, 2, 4, 3]))

'''
n = read_int()
target, value = [], []
for _ in range(n):
    line = read()
    t, v = line.split(' ')
    target.append(t)
    value.append(v)
write(sol.roc_auc(target=target, value=value))
'''