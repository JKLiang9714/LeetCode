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


class Solution63:
    """
    题意：给定一个m*n的矩阵，其中0代表可以走的路，1代表障碍物。机器人只能往下或往右走，初始位置在矩阵左上角，求可以让机器人走到矩阵右下角的路径的数量
    题解：dfs比较费时间，考虑动态规划如下：
        dp[0][0] = 0, if s[0][0] = 1
        dp[0][0] = 1, if s[0][0] = 0
        dp[i][j] = 0,                               if s[i][j] = 1
                 = dp[i - 1][j] + dp[i][j - 1],     if s[i][j] = 0
    """
    def uniquePathsWithObstacles(self, obstacleGrid):
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0 for i in range(n)] for j in range(m)]
        dp[0][0] = 0 if obstacleGrid[0][0] == 1 else 1
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] == 0
                else:
                    if i-1 >= 0:
                        dp[i][j] += dp[i-1][j]
                    if j-1 >= 0:
                        dp[i][j] += dp[i][j-1]
        return dp[m-1][n-1]


class Solution64:
    """
    题意：给定一个m*n的非负矩阵，矩阵中的数字代表权值，起点在矩阵左上角，只能往右或往下走，求走到矩阵右下角所需的最小路径长度
    题解：基本的动态规划题，dp[0][0]=grid[0][0]，dp[i][j]=grid[i][j]+min(dp[i-1][j],dp[i][j-1])，第一排和第一列由于没有上/左的格子，需要提前处理
    """
    def minPathSum(self, grid):
        m, n = len(grid), len(grid[0])
        for i in range(1, n):
            grid[0][i] += grid[0][i - 1]
        for i in range(1, m):
            grid[i][0] += grid[i - 1][0]
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        return grid[-1][-1]
        

class Solution66:
    """
    题意：给定一个非空的个位数数组，这个数组整体代表了一个非负整数。将这个非负整数+1后用个位数数组的形式返回
    """
    def plusOne(self, digits):
        num = 0
        for i in range(len(digits)):
            num += digits[i] * pow(10, len(digits) - i - 1)
        return [int(i) for i in str(num + 1)]


class Solution73:
    """
    题意：给定一个m*n的矩阵matrix，如果有一个元素是0，则将该元素的所在行和列都变为0。要求in-palce就地操作实现，也就是不使用临时变量，空间复杂度O(1)
    题解：利用matrix本身记录，首先定义row_flag和column_flag表示矩阵的第一行和第一列是否有0，然后扫描矩阵除了第一行和第一列以外的部分，用第一行和第一列置0来表示有0
    """
    def setZeroes(self, matrix):
        m, n = len(matrix), len(matrix[0])
        row_flag, col_flag = False, False
        for i in range(n):
            if matrix[0][i] == 0:
                row_flag = True
        for i in range(m):
            if matrix[i][0] == 0:
                col_flag = True
        for i in range(1,m):
            for j in range(1,n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        for i in range(1,m):
            for j in range(1,n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if row_flag:
            for i in range(n):
                matrix[0][i] = 0
        if col_flag:
            for i in range(m):
                matrix[i][0] = 0


class Solution74:
    """
    题意：给定一个m*n的整数矩阵，其中每行数从左到右升序排列，并且满足每行的第一个数大于上一行的最后一个数，给定一个target，确定target是否在这个矩阵中
    题解：在[0, row * col - 1]区间上二分，关键在于num = matrix[mid / cols][mid % cols]
    """
    def searchMatrix(self, matrix, target):
        if not matrix or target is None:
            return False
        rows, cols = len(matrix), len(matrix[0])
        low, high = 0, rows * cols - 1
        while low <= high:
            mid = int((low + high) / 2)
            num = matrix[int(mid / cols)][mid % cols]
            if num == target:
                return True
            elif num < target:
                low = mid + 1
            else:
                high = mid - 1
        return False


class Solution79:
    """
    题意：给定一个二维list和一个word，判断这个word是否能用二维list中相邻的字母连接而成(不能重复使用)
    题解：dfs，终止条件是当所有字母找完时返回True，当没找完并且四个方向都不能继续走下去时返回False。找到一个字母后分别向四个方向走，如果其中一个方向返回True则整体为True，走过的位置设为'#'，当四个方向都回来后将'#'重新变回原来的字母
    """
    def exist(self, board, word):
        if len(word) == 0:
            return False
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.dfs(board, i, j, word):
                    return True
        return False
    
    def dfs(self, board, i, j, word):
        if len(word) == 0:
            return True
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or word[0] != board[i][j]:
            return False
        tmp = board[i][j]
        board[i][j] = '#'
        res = self.dfs(board, i + 1, j, word[1:]) or self.dfs(board, i, j + 1, word[1:]) or self.dfs(board, i - 1, j, word[1:]) or self.dfs(board, i, j - 1, word[1:])
        board[i][j] = tmp
        return res


class Solution80:
    """
    题意：给定一个有序list，使得其中的数字不能重复出现两次以上，要求in-place做法，返回值为处理后的数组的长度
    题解：一次遍历
    """
    def removeDuplicates(self, nums):
        i = 0
        for n in nums:
            if i < 2 or n > nums[i - 2]:
                nums[i] = n
                i += 1
        return i


class Solution81:
    """
    题意：给定一个list，是由一个有序数组在某一枢纽处旋转得到的，并且其中可能含有重复元素，要求判断target是否在这个list中
    题解：虽然这个list经过旋转，但是还是可以用二分查找的思想，因为mid的左边或右边一定有一端是有序的。因此只需要在二分查找的时候对此进行判断就行了。另外本题可能有重复值，所以当left，mid和right指向的值都相等时要移动指针来跳出循环
    """
    def search(self, nums, target):
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = int((left + right) / 2)
            if nums[mid] == target:
                return True
            if nums[mid] < nums[right] or nums[mid] < nums[left]:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            elif nums[mid] > nums[left] or nums[mid] > nums[right]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                left += 1
        return False