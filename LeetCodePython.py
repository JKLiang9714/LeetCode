class Solution16:
    """
    题意：给定一个数组和一个目标值,找到数组中的三个数,使得这三个数之和与目标值之间的差距最小,返回它们的和
    题解：双指针，O(N^2)
    """

    def threeSumClosest(self, nums, target):
        nums.sort()
        result = nums[0] + nums[1] + nums[2]
        for i in range(len(nums) - 2):
            j, k = i + 1, len(nums) - 1
            while j < k:
                sum = nums[i] + nums[j] + nums[k]
                if abs(sum - target) < abs(result - target):
                    result = sum
                if sum == target:
                    return sum
                elif sum < target:
                    j += 1
                else:
                    k -= 1
        return result


class Solution18:
    """
    题意：找出list中所有相加等于target的4个数的list
    题解：先使用dict存储list中的两数之和和它们在list中的位置
          然后对于这个dict中的value,寻找一个key=target-value,然后将他们对应的数字存入list即可
    """

    def fourSum(self, nums, target):
        hash_dict = dict()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                sum = nums[i] + nums[j]
                if sum not in hash_dict:
                    hash_dict[sum] = [(i, j)]
                else:
                    hash_dict[sum].append((i, j))
        result = []
        for sum1 in hash_dict.keys():
            sum2 = target - sum1
            if sum2 in hash_dict.keys():
                for (i, j) in hash_dict[sum1]:
                    for (k, l) in hash_dict[sum2]:
                        if i != k and i != l and j != k and j != l:
                            r_list = [nums[i], nums[j], nums[k], nums[l]]
                            r_list.sort()
                            if r_list not in result:
                                result.append(r_list)
        return result


class Solution35:
    """
    题意：给定一个有序list和一个数target,求这个数在这个list中的位置,不存在时输出应该插入的位置
    题解：二分查找
    """

    def searchInsert(self, nums, target):
        return self.binarySearch(nums, target, 0, len(nums) - 1)

    def binarySearch(self, nums, target, l, r):
        while l <= r:
            mid = int((l + r) / 2)
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                return self.binarySearch(nums, target, mid + 1, r)
            else:
                return self.binarySearch(nums, target, l, mid - 1)
        return l


class Solution41:
    """
    题意：给定一个无序list,求出其中缺失的最小正整数,要求时间复杂度O(n)
    题解：
    """

    def firstMissingPositive(self, nums):
        if not nums:
            return 1
        nums_len = len(nums)
        pi = [0] * nums_len
        for i in range(nums_len):
            tmp = nums[i]
            if 1 <= tmp <= nums_len:
                pi[tmp - 1] = 1
        for i in range(nums_len):
            if pi[i] == 0:
                return i + 1
        return nums_len + 1


class Solution45:
    """
    题意：给定一个非负整数list，初始位置在list[0]，list的值代表了该位置能向前走的最大步数，求走到list末尾所需的最小次数(假设一定能够走到末尾)
    题解：既然要求最小次数，那么目的就是每一步都尽可能地往前走。这里的"尽可能"并非是每一步都走能走的最大步数，而是走到的位置加上该位置的最大步数，这代表了我们下一步的选择范围
    """

    def jump(self, nums):
        n, start, end, step = len(nums), 0, 0, 0
        while end < n - 1:
            step += 1
            max_end = end + 1
            for i in range(start, end + 1):
                if i + nums[i] >= n - 1:
                    return step
                max_end = max(max_end, i + nums[i])
            start, end = end + 1, max_end
        return step


class Solution59:
    """
    题意：给定一个正整数n，生成一个n*n的螺旋矩阵，其中元素从1一直到n^2
    题解：用di, dj巧妙地控制了四个方向，di, dj = dj, -di
    """
    def generateMatrix(self, n):
        matrix = [[0] * n for _ in range(n)]
        i, j, di, dj = 0, 0, 0, 1
        for k in range(n * n):
            matrix[i][j] = k + 1
            if matrix[(i+di) % n][(j+dj) % n]:
                di, dj = dj, -di
            i += di
            j += dj
        return matrix
        
