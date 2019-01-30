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
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        for i in range(1, m):
            for j in range(1, n):
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
        res = self.dfs(board, i + 1, j, word[1:]) or self.dfs(board, i, j + 1, word[1:]) or self.dfs(
            board, i - 1, j, word[1:]) or self.dfs(board, i, j - 1, word[1:])
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


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution95:
    """
    题意：给定一个数字n，生成所有存储了1~n的二叉查找树的可能形式
    题解：这题的思路是每次选取一个结点作为根，然后根据这个根把树切分为左右两个子树，再在左右两个子树里选取结点作为根，直至子树为空。
    注意子树为空时要返回[None]而不是[]，否则循环无法进行
    """

    def generateTrees(self, n):
        nums = list(range(1, n+1))
        if n == 0:
            return []
        return self.dfs(nums)

    def dfs(self, nums):
        if not nums:
            return [None]
        result = []
        for i in range(len(nums)):
            for l in self.dfs(nums[:i]):
                for r in self.dfs(nums[i+1:]):
                    node = TreeNode(nums[i])
                    node.left, node.right = l, r
                    result.append(node)
        return result


class Solution105:
    """
    题意：给定二叉树的前序遍历和中序遍历，输出该二叉树
    题解：前序遍历也就是根-左-右，中序遍历就是左-根-右。我们用递归的方式，preorder[0]必定是根结点,而这个根结点在inorder中的位置的左边是它的左子树，右边是它的右子树
    """

    def buildTree(self, preorder, inorder):
        if len(preorder) == 0:
            return None
        root_node = TreeNode(preorder[0])
        j = inorder.index(preorder[0])
        root_node.left = self.buildTree(preorder[1:j+1], inorder[0:j])
        root_node.right = self.buildTree(preorder[j+1:], inorder[j+1:])
        return root_node


class Solution106:
    """
    题意：给定二叉树的中序遍历和后序遍历，输出该二叉树
    题解：中序遍历是左-根-右，后序遍历是左-右-根。preorder[-1]必定是根结点，然后就和105题类似了
    """

    def buildTree(self, inorder, postorder):
        if len(postorder) == 0:
            return None
        root_node = TreeNode(postorder[-1])
        j = inorder.index(postorder[-1])
        root_node.left = self.buildTree(inorder[0:j], postorder[0:j])
        root_node.right = self.buildTree(inorder[j+1:], postorder[j:-1])
        return root_node


class Solution119:
    """
    题意：杨辉三角问题，给定k，要求输出杨辉三角的第k行
    题解：虽然看似是等腰三角形，但其实我们可以把它看做一个直角三角形，也就是矩阵的下半部分
    这题如果用O(k^2)的空间的话非常简单，抓住t[i][j] = t[i-1][j] + t[i-1][j-1]即可
    题干给了一个挑战，是用O(k)的空间完成，其实也非常简单，只要设置两个临时变量
    分别存储我们要修改的位置的上一层的这一位和前一位即可(逻辑上的上层，实际上只有一维数组)
    """

    def getRow(self, rowIndex):
        size = rowIndex + 1
        tri = [0] * size
        tri[0] = 1

        for _ in range(1, size):
            t1 = 1
            for j in range(1, size):
                t2 = tri[j]
                tri[j] += t1
                t1 = t2
        return tri


class Solution120:
    """
    题意：给定一个三角形的list，求出从顶到底的最短路径
    题解：经典DP题，算式为 tri[n-1][i] += min(tri[n][i], tri[n][i+1])
    """

    def minimumTotal(self, triangle):
        if not triangle:
            return
        for i in range(len(triangle)-2, -1, -1):
            for j in range(len(triangle[i])):
                triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1])
        return triangle[0][0]


class Solution152:
    """
    题意：给定一个list，找出其中一个连续的子数组，使得其中所有数的乘积最大
    题解：因为最小值乘负数会变最大，所以用两个变量存储当前最大值和最小值
    """

    def maxProduct(self, nums):
        maximum = big = small = nums[0]
        for n in nums[1:]:
            big, small = max(n, n*big, n*small), min(n, n*big, n*small)
            maximum = max(maximum, big)
        return maximum


class Solution153:
    """
    题意：给定一个排序过的list，在某一结点旋转过，找出其中的最小值
    题解：类似81题，还是用二分查找的思想。虽然list被旋转过，但是left-mid和mid-right其中的一段必定是有序的
    """

    def findMin(self, nums):
        left, right = 0, len(nums) - 1
        mid = (left + right) // 2
        while left < right:
            if nums[right] < nums[mid]:
                left = mid + 1
            else:
                right = mid
            mid = (left + right) // 2
        return min(nums[left], nums[right])


class Solution167:
    """
    题意：给定一个排序过的list，从中找到和等于target的两个数的位置，返回它们以1为起始值的坐标
    题解：双指针的思想，比较简单。值得一提的是，如果需要从无序数组中找到是否有两数之和等于某一target，也是采用先排序再双指针的方法
    """

    def twoSum(self, numbers, target):
        l, r = 0, len(numbers) - 1
        while l < r:
            if numbers[l] + numbers[r] > target:
                r -= 1
            elif numbers[l] + numbers[r] < target:
                l += 1
            else:
                break
        ans = [l+1, r+1]
        return ans


class Solution189:
    """
    题意：给定一个list和一个k，使这个list旋转k步
    题解：利用python的切片即可
    """

    def rotate(self, nums, k):
        l = len(nums)
        k %= l
        nums[:] = nums[l-k:] + nums[:l-k]


class Solution209:
    """
    题意：给定一个list和一个正数s，找到list中和大于等于s的最小连续区间的长度。如果没有则返回0
    题解：双指针法，用一个滑动的窗口去匹配，如果窗口内的值大于等于s则左移左边框，否则右移右边框，直到右边框到达数组底部并且窗口值小于s位置
    """

    def minSubArrayLen(self, s, nums):
        if sum(nums) < s:
            return 0
        elif max(nums) >= s:
            return 1
        l, r = 0, 1
        add = nums[l] + nums[r]
        min_windows = len(nums)

        while l < len(nums):
            if add >= s:
                min_windows = min(min_windows, r - l + 1)
                add -= nums[l]
                l += 1
            else:
                if r < len(nums)-1:
                    r += 1
                    add += nums[r]
                else:
                    break
        return min_windows


class Solution216:
    """
    题意：给定一个数k和一个数n，要求找到1-9内的k个数，且满足它们的和为n的所有可能组合
    题解：
    回溯法：保存当前步骤，如果是一个解就输；维护状态，使搜索路径（含子路径）尽量不重复。必要时，应该对不可能为解的部分进行剪枝
    递归函数的开头写好跳出条件，满足条件才将当前结果加入总结果中
    已经拿过的数不再拿
    遍历过当前结点后，为了回溯到上一步，要去掉已经加入到结果list中的当前结点
    代入到这题中，每一个dfs都遍历1-9中当前index后面的数，这确保了已经拿过的数不再拿。进入下一层dfs，并令k-1，n-nums[index]，跳出条件是k<0或n<0，满足条件是k==0且n==0
    """

    def combinationSum3(self, k, n):
        res = []
        self.dfs(range(1, 10), k, n, 0, [], res)
        return res

    def dfs(self, nums, k, n, index, path, res):
        if k < 0 or n < 0:  # backtracking
            return
        if k == 0 and n == 0:
            res.append(path)
        for i in range(index, len(nums)):
            self.dfs(nums, k-1, n-nums[i], i+1, path+[nums[i]], res)


class Solution228:
    """
    题意：给定一个有序且无重复数字的list，将其中连续范围的数字合并后返回
    题解：根据题意，我们需要确认的其实就是每段连续区间的首尾数字。首数字可能是list的第一个数或是前一个数和它不连续的数，尾数字可能是list的最后一个数或是后一个数和它不连续的数。并且每一个尾数字一定对应着一段连续区间，将这段区间存入一个字符list即可
    """

    def summaryRanges(self, nums):
        summary = []
        start = 0
        end = 0
        for i in range(len(nums)):
            if i == 0 or nums[i-1]+1 != nums[i]:
                start = nums[i]
            if i == len(nums)-1 or nums[i+1]-1 != nums[i]:
                end = nums[i]
                if start == end:
                    summary.append(str(start))
                else:
                    summary.append(str(start) + '->' + str(end))
        return summary


class Solution229:
    """
    题意：给定一个长度为n的list，找到其中出现次数大于[n/3]的所有数。要求时间复杂度O(n)，空间复杂度O(1)
    题解：
    使用dict存储这个list中每个数出现的次数，然后将其中次数大于[n/3]的存入一个list。但是则不符合空间复杂度的要求
    查阅solution后发现这题可以使用Boyer-Moore多数投票算法解决。这是一种寻找"多数元素"的好方法，基本思想是建立标志位和count，如果匹配到的数字不等于标志位则让count-1，否则count+1，如果count为0时更换标志位。因为本题要求的是出现次数大于[n/3]的所有数，也就是最多可能有两个数，因此要建立两组标志位和count
    """

    def majorityElement(self, nums):
        count1, count2, tmp1, tmp2 = 0, 0, 0, 1
        for i in nums:
            if i == tmp1:
                count1 += 1
            elif i == tmp2:
                count2 += 1
            elif count1 == 0:
                tmp1 = i
                count1 = 1
            elif count2 == 0:
                tmp2 = i
                count2 = 1
            else:
                count1 -= 1
                count2 -= 1

        ans = [n for n in (tmp1, tmp2) if nums.count(n) > len(nums) // 3]
        return ans
