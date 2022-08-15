from typing import List
import numpy as np
from heapq import heappush, heappop
from bisect import bisect, insort
from sys import stdin, stdout
from functools import wraps
from time import time
import tracemalloc


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print(f"time : {te-ts}")
        return result

    return wrap


def read():
    return stdin.readline().rstrip("\n")


def read_int():
    return int(read())


def write(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    stdout.write(sep.join(str(a) for a in args) + end)


class MergeSort:
    @classmethod
    def mergeSort(cls, arr, n):
        temp_arr = [0] * n
        return cls._mergeSort(arr, temp_arr, 0, n - 1)

    @classmethod
    def _mergeSort(cls, arr, temp_arr, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            inv_count += cls._mergeSort(arr, temp_arr, left, mid)
            inv_count += cls._mergeSort(arr, temp_arr, mid + 1, right)
            inv_count += cls.merge(arr, temp_arr, left, mid, right)
        return inv_count

    @classmethod
    def merge(cls, arr, temp_arr, left, mid, right):
        i = left    
        j = mid + 1  
        k = left     
        inv_count = 0

        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp_arr[k] = arr[i]
                k += 1
                i += 1
            else:
                temp_arr[k] = arr[j]
                inv_count += (mid-i + 1)
                k += 1
                j += 1

        while i <= mid:
            temp_arr[k] = arr[i]
            k += 1
            i += 1

        while j <= right:
            temp_arr[k] = arr[j]
            k += 1
            j += 1
    
        for loop_var in range(left, right + 1):
            arr[loop_var] = temp_arr[loop_var]
    
        return inv_count

class Solution:
    def __init__(self):
        ...

    @classmethod
    # @timing
    def roc_auc(cls, target: List[float], value: List[float]) -> float:
        """
        Calculate ROCAUC

        find number inversion in index_sorted_by_value_target
        """
        index_sorted_value = np.argsort(value)
        sorted_target = np.sort(target)
        sorted_value = np.array(value)[index_sorted_value]
        sorted_by_value_target = np.array(target)[index_sorted_value]
        sorted_by_value_target = cls._sort_target_in_each_equal_range_value(
            sorted_value, sorted_by_value_target
        )

        deb0 = cls._number_of_non_inversions(sorted_by_value_target)
        deb2 = cls._number_of_non_inversions(sorted_target)
        deb3 = cls._count_combinations_of_pairs(sorted_target)
        deb4 = cls._count_equal_target(sorted_value, sorted_by_value_target)

        numerator = deb0 - deb3 - deb4 / 2
        denominator = deb2 - deb3

        return numerator / denominator

    @classmethod
    def roc_auc_less_memory(cls, target: List[float], value: List[float]) -> float:
        """
        Calculate ROCAUC

        find number inversion in index_sorted_by_value_target
        """
        index_sorted_value = np.argsort(value)
        sorted_value = np.array(value)[index_sorted_value]
        sorted_by_value_target = np.array(target)[index_sorted_value]
        sorted_by_value_target = cls._sort_target_in_each_equal_range_value(
            sorted_value, sorted_by_value_target
        )

        mergeSort = MergeSort()

        deb4 = cls._count_equal_target(sorted_value, sorted_by_value_target)
        deb0 = mergeSort.mergeSort(sorted_by_value_target, len(sorted_by_value_target))
        deb2 = int(cls._number_of_pairs(sorted_by_value_target))
        deb3 = cls._count_combinations_of_pairs(sorted_by_value_target)
        

        numerator = deb2 - deb0 - deb3 - deb4 / 2
        denominator = deb2 - deb3

        return numerator / denominator

    @classmethod
    def _number_of_pairs(cls, arr: List[float]) -> int:
        return len(arr)*(len(arr) - 1)/2

    @classmethod
    def _sort_target_in_each_equal_range_value(cls, sorted_value, sorted_target):
        right = 0
        left = 0
        res = [-1 for _ in range(len(sorted_target))]
        while True:
            while sorted_value[right] == sorted_value[left]:
                right += 1
                if right == len(sorted_value):
                    res[left:right] = np.sort(sorted_target[left:right])
                    return res
            res[left:right] = np.sort(sorted_target[left:right])
            left = right

    @classmethod
    def _count_combinations_of_pairs(cls, sorted_arr: List[float]) -> int:
        res = 0
        right = 0
        left = 0
        while True:
            while sorted_arr[right] == sorted_arr[left]:
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

        for i, v in enumerate(arr):
            heappush(sortList, (v, i))

        x = []
        while sortList:
            v, i = heappop(sortList)
            y = bisect(x, i)
            result += y

            insort(x, i)

        return result

    @classmethod
    def _number_of_non_inversions_in_equal_target(cls, arr: List[float]) -> int:
        try:
            arr = arr
            return cls._number_of_non_inversions(
                arr
            ) - cls._count_combinations_of_pairs(arr)
        except:
            return 0

    @classmethod
    def _count_equal_target(
        cls, _sorted_value: List[float], _sorted_by_value_target: List[float]
    ) -> int:
        res = 0
        right = 0
        left = 0
        while True:
            while _sorted_value[right] == _sorted_value[left]:
                right += 1
                if right == len(_sorted_value):
                    res += cls._number_of_non_inversions_in_equal_target(
                        _sorted_by_value_target[left:right]
                    )
                    return int(res)
            res += cls._number_of_non_inversions_in_equal_target(
                _sorted_by_value_target[left:right]
            )
            left = right


sol = Solution()
"""
print(sol.roc_auc(target=[0, 1], value=[0, 1])) 
print(sol.roc_auc(target=[0, 1], value=[1, 0]))
print(sol.roc_auc(target=[0.5, 0.5, 2], value=[0, 1, 0.5]))
print(sol.roc_auc(target=[0, 0, 1, 0, 1, 1, 1], value=[0, 1, 2, 3, 4, 5, 6]))
print(sol.roc_auc(target=[0, 3, 1, 2, 1, 2, 4, 2, 4, 0], value=[4, 0, 2, 4, 0, 1, 1, 1, 4, 0]))
print(sol.roc_auc(target=[3, 3, 2, 3, 4, 0, 2, 1, 4, 3], value=[4, 4, 1, 2, 1, 2, 0, 2, 4, 3]))
"""

'''
arr1 = [4, 0, 2, 4, 0, 1, 1, 1, 4, 0]
arr2 = [4, 0, 2, 4, 0, 1, 1, 1, 4, 0]
mergeSort = MergeSort
res = mergeSort.mergeSort(arr1, len(arr1))
all_comb = (len(arr1))*(len(arr1) - 1)/2
print(all_comb - res, arr1)
print(sol._number_of_non_inversions(arr2))
'''


DEBUG = True

if not DEBUG:
    n = read_int()
    target, value = [], []
    for _ in range(n):
        line = read()
        t, v = line.split(" ")
        target.append(t)
        value.append(v)
    write(sol.roc_auc(target=target, value=value))
else:
    #tracemalloc.start()
    with open("input.txt", "r") as input_file:
        ts = time()
        n = input_file.readline()
        target, value = [], []
        for _ in range(int(n)):
            line = input_file.readline()
            t, v = line.split(" ")
            target.append(t)
            value.append(v)
        with open("output.txt", "w") as output_file:
            #output_file.write(str(sol.roc_auc(target=target, value=value)))
            output_file.write(str(sol.roc_auc_less_memory(target=target, value=value)))
        # write(sol.roc_auc(target=target, value=value))
        write(time() - ts)
        #write(tracemalloc.get_traced_memory())
        #tracemalloc.stop()
